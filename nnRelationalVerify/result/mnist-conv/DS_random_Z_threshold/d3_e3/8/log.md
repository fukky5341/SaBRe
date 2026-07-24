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
execution time: IAR + RelationalAnalysis = 21.31 + 34.85 = 56.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.7198690, upper bound: 0.7198687

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6192
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6192

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7142753
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142740, upper bound: 0.7198651
time: 5.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.26 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.26
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7142753
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.26
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

Time for backsubstitution: 20.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198634, upper bound: 0.7126314
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7182524, upper bound: 0.7142735
time: 3.25 seconds

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

Time for backsubstitution: 20.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142739, upper bound: 0.7174553
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7118736, upper bound: 0.7198651
time: 3.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.16
Output dim: 7, lower bound: -0.7198634, upper bound: 0.7126314
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.16
Output dim: 7, lower bound: -0.7182524, upper bound: 0.7142735
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.16
Output dim: 7, lower bound: -0.7142739, upper bound: 0.7174553
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.16
Output dim: 7, lower bound: -0.7118736, upper bound: 0.7198651

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5720840, 1.5781863
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7921534, 1.7950587
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6073623, 1.6093712
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4529614, 1.4543293
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2093799, 1.2038553
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3940954, 1.3938105
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5669570, 1.5659585
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2107742, 1.2081661
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2880611, 1.2923944
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4362688, 1.4356332

Time for backsubstitution: 21.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184425, upper bound: 0.7126228
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198572, upper bound: 0.7111848
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5715766, 1.5786934
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7919378, 1.7952743
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6091695, 1.6075644
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4562883, 1.4510021
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2028196, 1.2104154
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3949380, 1.3929677
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5695701, 1.5633457
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2106755, 1.2082644
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2896118, 1.2908432
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4402957, 1.4316068

Time for backsubstitution: 21.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7168138, upper bound: 0.7142649
time: 8.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7182462, upper bound: 0.7128272
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5790272, 1.5724177
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7954140, 1.7922940
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6105285, 1.6103258
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4564848, 1.4584436
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2146783, 1.2136421
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3943548, 1.3954830
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5676374, 1.5712500
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2083268, 1.2108359
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2933993, 1.2906168
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4382367, 1.4428985

Time for backsubstitution: 19.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142743, upper bound: 0.7174297
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7118495, upper bound: 0.7174555
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5790272, 1.5724177
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7954140, 1.7922935
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6105280, 1.6103263
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4564843, 1.4584439
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2146773, 1.2136431
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3943558, 1.3954825
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5676389, 1.5712490
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2083268, 1.2108364
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2933993, 1.2906165
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4382362, 1.4428992

Time for backsubstitution: 19.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7104214, upper bound: 0.7198592
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7118673, upper bound: 0.7184466
time: 3.61 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.98 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7184425, upper bound: 0.7126228
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7198572, upper bound: 0.7111848
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7168138, upper bound: 0.7142649
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7182462, upper bound: 0.7128272
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7142743, upper bound: 0.7174297
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7118495, upper bound: 0.7174555
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7104214, upper bound: 0.7198592
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.98
Output dim: 7, lower bound: -0.7118673, upper bound: 0.7184466

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

Time for backsubstitution: 20.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102164
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7159950, upper bound: 0.7126230
time: 3.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 20.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087720
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7111829
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 20.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7168138, upper bound: 0.7118650
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7144079, upper bound: 0.7142672
time: 3.80 seconds

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

Time for backsubstitution: 20.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7182462, upper bound: 0.7104209
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7158435, upper bound: 0.7128279
time: 3.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5722814, 1.5671573
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7815237, 1.7807193
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.5999227, 1.5976000
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4503117, 1.4510362
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2137833, 1.2125652
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3891602, 1.3915899
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5470629, 1.5541072
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.1974714, 1.1974072
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2918463, 1.2887545
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4441619, 1.4453878

Time for backsubstitution: 20.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7128302, upper bound: 0.7174236
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7142680, upper bound: 0.7159960
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5737667, 1.5656722
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7838392, 1.7784047
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.5978026, 1.5997200
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4490771, 1.4522703
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2136011, 1.2127471
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3904619, 1.3902879
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5504947, 1.5506756
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.1948979, 1.1999803
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2915368, 1.2890635
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4407258, 1.4488237

Time for backsubstitution: 20.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7118470, upper bound: 0.7158497
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7102235, upper bound: 0.7174535
time: 3.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5751600, 1.5662704
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7901869, 1.7890081
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6071401, 1.6049218
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4556522, 1.4571161
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2101564, 1.2064164
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3942595, 1.3953292
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5604610, 1.5667496
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2073150, 1.2102005
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2922716, 1.2888176
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4264264, 1.4354844

Time for backsubstitution: 20.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7104217, upper bound: 0.7174247
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7103960, upper bound: 0.7198597
time: 3.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5728798, 1.5685506
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7921276, 1.7870665
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6051235, 1.6069384
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4551568, 1.4576118
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2074513, 1.2091215
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3942022, 1.3953867
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5631394, 1.5640712
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2076907, 1.2098246
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2916002, 1.2894886
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4308209, 1.4310894

Time for backsubstitution: 20.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7118676, upper bound: 0.7159969
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7118419, upper bound: 0.7184446
time: 3.62 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102164
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7159950, upper bound: 0.7126230
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087720
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7111829
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7168138, upper bound: 0.7118650
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7144079, upper bound: 0.7142672
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7182462, upper bound: 0.7104209
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7158435, upper bound: 0.7128279
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7128302, upper bound: 0.7174236
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7142680, upper bound: 0.7159960
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7118470, upper bound: 0.7158497
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7102235, upper bound: 0.7174535
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7104217, upper bound: 0.7174247
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7103960, upper bound: 0.7198597
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7118676, upper bound: 0.7159969
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 7, lower bound: -0.7118419, upper bound: 0.7184446

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

Time for backsubstitution: 20.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102157
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7160196, upper bound: 0.7102162
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 20.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087710
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7174474, upper bound: 0.7087743
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5606761, 1.5675735
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7772923, 1.7759409
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.5892315, 1.5953770
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4442263, 1.4473236
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2010732, 1.1984351
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3900456, 1.3885164
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5453153, 1.5382071
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.1967077, 1.1962974
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2844000, 1.2897139
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4313374, 1.4297431

Time for backsubstitution: 20.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7087967
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7174217, upper bound: 0.7111829
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5677094, 1.5725458
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7867098, 1.7919865
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6057782, 1.6021562
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4554524, 1.4496701
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1982958, 1.2031851
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3948393, 1.3928123
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5623860, 1.5588412
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2096624, 1.2076271
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2884860, 1.2890465
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4284830, 1.4241881

Time for backsubstitution: 20.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7168141, upper bound: 0.7118397
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7143835, upper bound: 0.7118672
time: 3.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5654292, 1.5748258
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7886515, 1.7900453
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6037612, 1.6041732
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4549570, 1.4501657
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1955903, 1.2058902
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3947821, 1.3928699
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5650644, 1.5561628
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2100387, 1.2072513
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2878146, 1.2897177
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4328775, 1.4197936

Time for backsubstitution: 20.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7182465, upper bound: 0.7103955
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7158192, upper bound: 0.7104183
time: 3.98 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7160196, upper bound: 0.7102162
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087710
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7174474, upper bound: 0.7087743
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7087967
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7174217, upper bound: 0.7111829
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7168141, upper bound: 0.7118397
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7143835, upper bound: 0.7118672
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7182465, upper bound: 0.7103955
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 7, lower bound: -0.7158192, upper bound: 0.7104183
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 7, lower bound: -0.7128302, upper bound: 0.7174236
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 7, lower bound: -0.7102235, upper bound: 0.7174535
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 7, lower bound: -0.7104217, upper bound: 0.7174247
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 7, lower bound: -0.7103960, upper bound: 0.7198597
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.49
Output dim: 7, lower bound: -0.7118419, upper bound: 0.7184446

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.16 + 543.93 = 600.09 seconds
