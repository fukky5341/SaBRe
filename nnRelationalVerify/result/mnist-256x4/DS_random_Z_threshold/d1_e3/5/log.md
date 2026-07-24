## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000711535


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0031298, 0.0031298)
1: (-0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008824, 0.0008824)
2: (0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0065105, 0.0065105)
3: (0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008616, 0.0008616)
4: (-0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048656, 0.0048656)
5: (0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013518, 0.0013518)
6: (0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012270, 0.0012270)
7: (-0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045790, 0.0045790)
8: (-0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035639, 0.0035639)
9: (-0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003075, 0.0003075)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.04 + 2.91 = 3.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0008371, upper bound: 0.0008363

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008159, upper bound: 0.0008172
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008159, upper bound: 0.0008159
time: 1.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.14 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.14
Output dim: 5, lower bound: -0.0008159, upper bound: 0.0008172
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.14
Output dim: 5, lower bound: -0.0008159, upper bound: 0.0008159

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030575, 0.0030540
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008620, 0.0008610
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063602, 0.0063529
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008417, 0.0008407
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047478, 0.0047532
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013191, 0.0013206
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011973, 0.0011987
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044682, 0.0044733
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034816, 0.0034776
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003000, 0.0003004

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007860, upper bound: 0.0007992
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007860, upper bound: 0.0007897
time: 2.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030540, 0.0030575
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008610, 0.0008620
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063529, 0.0063602
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008407, 0.0008417
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047532, 0.0047478
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013206, 0.0013191
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011987, 0.0011973
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044733, 0.0044682
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034776, 0.0034816
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003004, 0.0003000

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008126, upper bound: 0.0008119
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008126, upper bound: 0.0008126
time: 1.90 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 5, lower bound: -0.0007860, upper bound: 0.0007992
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 5, lower bound: -0.0007860, upper bound: 0.0007897
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 5, lower bound: -0.0008126, upper bound: 0.0008119
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.78
Output dim: 5, lower bound: -0.0008126, upper bound: 0.0008126

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029871, 0.0029726
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008422, 0.0008381
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062137, 0.0061837
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008223, 0.0008183
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046213, 0.0046437
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012839, 0.0012902
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011654, 0.0011711
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043491, 0.0043703
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034014, 0.0033849
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002920, 0.0002935

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007826, upper bound: 0.0007959
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007826, upper bound: 0.0007958
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029761, 0.0029830
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008391, 0.0008410
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061910, 0.0062052
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008193, 0.0008212
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046374, 0.0046267
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012884, 0.0012854
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011695, 0.0011668
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043643, 0.0043543
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033889, 0.0033967
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002931, 0.0002924

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007363, upper bound: 0.0007404
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007363, upper bound: 0.0007633
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030660, 0.0030648
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008644, 0.0008641
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063779, 0.0063754
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008440, 0.0008437
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047646, 0.0047664
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013237, 0.0013243
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012016, 0.0012020
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044840, 0.0044857
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034913, 0.0034899
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003011, 0.0003012

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007612, upper bound: 0.0007610
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007612, upper bound: 0.0007856
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030613, 0.0030714
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008631, 0.0008659
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063681, 0.0063891
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008427, 0.0008455
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047748, 0.0047591
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013266, 0.0013222
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012041, 0.0012002
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044936, 0.0044789
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034859, 0.0034974
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003017, 0.0003007

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008070, upper bound: 0.0008062
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008070, upper bound: 0.0008072
time: 1.93 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.94 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0007826, upper bound: 0.0007959
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0007826, upper bound: 0.0007958
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0007363, upper bound: 0.0007404
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0007363, upper bound: 0.0007633
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0007612, upper bound: 0.0007610
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0007612, upper bound: 0.0007856
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0008070, upper bound: 0.0008062
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0008070, upper bound: 0.0008072

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030007, 0.0029814
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008460, 0.0008406
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062422, 0.0062019
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008261, 0.0008207
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046349, 0.0046650
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012877, 0.0012961
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011689, 0.0011764
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043620, 0.0043903
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034170, 0.0033949
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002929, 0.0002948

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007805, upper bound: 0.0007941
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007805, upper bound: 0.0007931
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029958, 0.0029861
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008446, 0.0008419
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062319, 0.0062116
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008247, 0.0008220
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046422, 0.0046574
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012897, 0.0012940
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011707, 0.0011745
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043688, 0.0043831
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034114, 0.0034003
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002934, 0.0002943

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007534, upper bound: 0.0007601
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007459, upper bound: 0.0007670
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029047, 0.0029556
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008190, 0.0008333
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060424, 0.0061482
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007996, 0.0008136
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045948, 0.0045157
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012766, 0.0012546
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011587, 0.0011388
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043242, 0.0042498
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033076, 0.0033655
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002904, 0.0002854

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007257, upper bound: 0.0007324
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007257, upper bound: 0.0007399
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029506, 0.0029116
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008319, 0.0008209
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061378, 0.0060566
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008122, 0.0008015
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045264, 0.0045870
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012576, 0.0012744
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011415, 0.0011568
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042598, 0.0043169
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033598, 0.0033154
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002860, 0.0002899

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007217, upper bound: 0.0007507
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007217, upper bound: 0.0007483
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029950, 0.0030396
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008444, 0.0008570
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062302, 0.0063231
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008245, 0.0008368
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047255, 0.0046560
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013129, 0.0012936
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011917, 0.0011742
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044472, 0.0043819
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034104, 0.0034613
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002986, 0.0002942

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007867, upper bound: 0.0007505
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007508, upper bound: 0.0007602
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030396, 0.0029938
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008570, 0.0008441
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063230, 0.0062277
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008367, 0.0008241
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046542, 0.0047254
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012931, 0.0013129
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011737, 0.0011917
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043801, 0.0044471
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034612, 0.0034091
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002941, 0.0002986

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007508, upper bound: 0.0007701
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007508, upper bound: 0.0007844
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030509, 0.0030626
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008601, 0.0008635
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063464, 0.0063708
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008398, 0.0008431
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047611, 0.0047429
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013228, 0.0013177
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012007, 0.0011961
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044808, 0.0044636
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034740, 0.0034874
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003009, 0.0002997

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007867, upper bound: 0.0007819
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007827, upper bound: 0.0007832
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030525, 0.0030616
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008606, 0.0008632
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063498, 0.0063687
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008403, 0.0008428
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047595, 0.0047455
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013223, 0.0013184
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012003, 0.0011967
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044793, 0.0044660
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034759, 0.0034862
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003008, 0.0002999

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007722, upper bound: 0.0007593
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007620, upper bound: 0.0007689
time: 1.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007805, upper bound: 0.0007941
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007805, upper bound: 0.0007931
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007534, upper bound: 0.0007601
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007459, upper bound: 0.0007670
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007257, upper bound: 0.0007324
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007257, upper bound: 0.0007399
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007217, upper bound: 0.0007507
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007217, upper bound: 0.0007483
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007867, upper bound: 0.0007505
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007508, upper bound: 0.0007602
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007508, upper bound: 0.0007701
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007508, upper bound: 0.0007844
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007867, upper bound: 0.0007819
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007827, upper bound: 0.0007832
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007722, upper bound: 0.0007593
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 5, lower bound: -0.0007620, upper bound: 0.0007689

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030012, 0.0029808
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008461, 0.0008404
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062431, 0.0062007
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008262, 0.0008206
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046340, 0.0046657
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012875, 0.0012963
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011686, 0.0011766
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043611, 0.0043909
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034175, 0.0033943
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002928, 0.0002948

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007754, upper bound: 0.0007890
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007754, upper bound: 0.0007923
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030002, 0.0029817
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008459, 0.0008407
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062410, 0.0062026
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008259, 0.0008208
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046354, 0.0046641
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012879, 0.0012958
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011690, 0.0011762
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043625, 0.0043894
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034163, 0.0033953
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002929, 0.0002947

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007654, upper bound: 0.0007803
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007654, upper bound: 0.0007761
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029520, 0.0029629
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008323, 0.0008353
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061408, 0.0061634
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008126, 0.0008156
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046061, 0.0045892
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012797, 0.0012750
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011616, 0.0011573
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043349, 0.0043190
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033615, 0.0033739
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002911, 0.0002900

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007264
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007153, upper bound: 0.0007267
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029958, 0.0029422
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008446, 0.0008295
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062319, 0.0061205
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008247, 0.0008099
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045741, 0.0046574
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012708, 0.0012940
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011535, 0.0011745
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043047, 0.0043831
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034114, 0.0033504
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002891, 0.0002943

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007433, upper bound: 0.0007648
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007433, upper bound: 0.0007629
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0028629, 0.0029260
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008072, 0.0008249
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0059554, 0.0060867
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007881, 0.0008055
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045488, 0.0044507
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012638, 0.0012365
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011471, 0.0011224
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042809, 0.0041886
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0032600, 0.0033319
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002875, 0.0002813

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007231, upper bound: 0.0007298
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007231, upper bound: 0.0007294
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0028752, 0.0029113
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008106, 0.0008208
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0059809, 0.0060561
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007915, 0.0008014
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045259, 0.0044698
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012574, 0.0012418
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011414, 0.0011272
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042594, 0.0042066
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0032740, 0.0033151
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002860, 0.0002825

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007113, upper bound: 0.0006913
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007078, upper bound: 0.0007004
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027474, 0.0027083
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007746, 0.0007636
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057151, 0.0056338
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007563, 0.0007455
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042104, 0.0042711
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011698, 0.0011866
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010618, 0.0010771
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039624, 0.0040196
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031284, 0.0030840
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002661, 0.0002699

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007170, upper bound: 0.0007475
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007170, upper bound: 0.0007486
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027473, 0.0027113
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007746, 0.0007644
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057149, 0.0056401
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007563, 0.0007464
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042150, 0.0042710
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011711, 0.0011866
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010630, 0.0010771
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039668, 0.0040195
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031284, 0.0030874
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002664, 0.0002699

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0006967
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0007040
time: 2.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029534, 0.0030116
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008327, 0.0008491
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061437, 0.0062648
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008130, 0.0008291
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046819, 0.0045914
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013008, 0.0012756
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011807, 0.0011579
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044062, 0.0043210
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033631, 0.0034294
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002959, 0.0002901

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007291, upper bound: 0.0007282
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007283, upper bound: 0.0007288
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029670, 0.0029987
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008365, 0.0008455
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061719, 0.0062380
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008168, 0.0008255
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046619, 0.0046125
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012952, 0.0012815
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011757, 0.0011632
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043873, 0.0043409
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033785, 0.0034147
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002946, 0.0002915

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007275, upper bound: 0.0007297
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0007323
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029974, 0.0029658
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008451, 0.0008362
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062352, 0.0061695
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008251, 0.0008164
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046107, 0.0046598
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012810, 0.0012946
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011627, 0.0011751
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043392, 0.0043854
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034132, 0.0033772
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002914, 0.0002945

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007468
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007463
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030116, 0.0029535
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008491, 0.0008327
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062647, 0.0061439
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008290, 0.0008130
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045916, 0.0046818
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012757, 0.0013008
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011579, 0.0011807
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043212, 0.0044061
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034293, 0.0033632
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002902, 0.0002959

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007291, upper bound: 0.0007597
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007283, upper bound: 0.0007609
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030386, 0.0030521
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008567, 0.0008605
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063209, 0.0063490
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008365, 0.0008402
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047449, 0.0047238
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013183, 0.0013124
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011966, 0.0011913
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044654, 0.0044457
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034601, 0.0034755
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002998, 0.0002985

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007436, upper bound: 0.0007408
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007425, upper bound: 0.0007423
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030414, 0.0030503
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008575, 0.0008600
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063267, 0.0063453
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008372, 0.0008397
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047421, 0.0047282
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013175, 0.0013136
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011959, 0.0011924
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044628, 0.0044497
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034632, 0.0034734
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002997, 0.0002988

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007713, upper bound: 0.0007711
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007713, upper bound: 0.0007821
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030147, 0.0030470
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008499, 0.0008591
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062711, 0.0063384
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008299, 0.0008388
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047369, 0.0046866
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013161, 0.0013021
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011946, 0.0011819
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044580, 0.0044106
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034328, 0.0034696
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002993, 0.0002962

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007710, upper bound: 0.0007487
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007549, upper bound: 0.0007578
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030525, 0.0030237
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008606, 0.0008525
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063498, 0.0062900
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008403, 0.0008324
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047007, 0.0047455
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013060, 0.0013184
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011855, 0.0011967
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044239, 0.0044660
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034759, 0.0034431
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002971, 0.0002999

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007296, upper bound: 0.0007340
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007296, upper bound: 0.0007332
time: 2.00 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007754, upper bound: 0.0007890
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007754, upper bound: 0.0007923
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007654, upper bound: 0.0007803
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007654, upper bound: 0.0007761
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007264
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007153, upper bound: 0.0007267
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007433, upper bound: 0.0007648
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007433, upper bound: 0.0007629
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007231, upper bound: 0.0007298
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007231, upper bound: 0.0007294
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007113, upper bound: 0.0006913
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007078, upper bound: 0.0007004
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007170, upper bound: 0.0007475
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007170, upper bound: 0.0007486
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0006967
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0007040
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007291, upper bound: 0.0007282
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007283, upper bound: 0.0007288
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007275, upper bound: 0.0007297
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0007323
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007468
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007463
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007291, upper bound: 0.0007597
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007283, upper bound: 0.0007609
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007436, upper bound: 0.0007408
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007425, upper bound: 0.0007423
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007713, upper bound: 0.0007711
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007713, upper bound: 0.0007821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007710, upper bound: 0.0007487
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007549, upper bound: 0.0007578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007296, upper bound: 0.0007340
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 5, lower bound: -0.0007296, upper bound: 0.0007332

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030012, 0.0029815
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008462, 0.0008406
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062431, 0.0062022
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008262, 0.0008208
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046352, 0.0046657
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012878, 0.0012963
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011689, 0.0011766
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043622, 0.0043910
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034175, 0.0033951
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002929, 0.0002948

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007593, upper bound: 0.0007744
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007593, upper bound: 0.0007740
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030019, 0.0029807
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008464, 0.0008404
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062446, 0.0062005
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008264, 0.0008205
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046338, 0.0046668
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012874, 0.0012966
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011686, 0.0011769
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043610, 0.0043920
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034183, 0.0033941
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002928, 0.0002949

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007658, upper bound: 0.0007804
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007658, upper bound: 0.0007911
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027864, 0.0027661
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007856, 0.0007799
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057964, 0.0057541
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007671, 0.0007615
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0043002, 0.0043319
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011947, 0.0012035
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010845, 0.0010924
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040470, 0.0040768
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031729, 0.0031498
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002717, 0.0002737

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007299, upper bound: 0.0007323
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007178, upper bound: 0.0007449
time: 2.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027846, 0.0027665
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007851, 0.0007800
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057925, 0.0057548
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007665, 0.0007616
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0043008, 0.0043289
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011949, 0.0012027
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010846, 0.0010917
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040475, 0.0040740
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031708, 0.0031502
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002718, 0.0002736

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007487, upper bound: 0.0007583
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007483, upper bound: 0.0007588
time: 2.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029236, 0.0029438
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008243, 0.0008300
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060817, 0.0061237
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008048, 0.0008104
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045765, 0.0045451
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012715, 0.0012628
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011541, 0.0011462
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043070, 0.0042774
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033291, 0.0033521
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002892, 0.0002872

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006545, upper bound: 0.0006646
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006545, upper bound: 0.0006646
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029324, 0.0029345
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008267, 0.0008273
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060999, 0.0061043
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008072, 0.0008078
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045620, 0.0045587
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012675, 0.0012665
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011505, 0.0011496
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042934, 0.0042902
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033391, 0.0033415
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002883, 0.0002881

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006874, upper bound: 0.0006990
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006874, upper bound: 0.0006990
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029962, 0.0029415
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008447, 0.0008293
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062327, 0.0061190
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008248, 0.0008098
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045729, 0.0046580
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012705, 0.0012941
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011532, 0.0011747
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043037, 0.0043837
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034118, 0.0033495
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002890, 0.0002944

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007175, upper bound: 0.0007369
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007175, upper bound: 0.0007373
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029953, 0.0029425
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008445, 0.0008296
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062307, 0.0061210
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008245, 0.0008100
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045744, 0.0046565
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012709, 0.0012937
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011536, 0.0011743
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043051, 0.0043823
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034107, 0.0033506
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002891, 0.0002943

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007353, upper bound: 0.0007498
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007353, upper bound: 0.0007502
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0028621, 0.0029243
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008069, 0.0008245
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0059537, 0.0060831
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007879, 0.0008050
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045462, 0.0044495
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012631, 0.0012362
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011465, 0.0011221
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042784, 0.0041874
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0032591, 0.0033299
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002873, 0.0002812

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006989, upper bound: 0.0007006
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0007054
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0028612, 0.0029256
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008067, 0.0008248
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0059518, 0.0060858
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007876, 0.0008054
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045481, 0.0044480
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012636, 0.0012358
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011470, 0.0011217
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042803, 0.0041861
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0032580, 0.0033314
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002874, 0.0002811

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006990, upper bound: 0.0007005
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0007054
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027503, 0.0027122
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007754, 0.0007647
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057212, 0.0056418
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007571, 0.0007466
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042164, 0.0042756
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011714, 0.0011879
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010633, 0.0010783
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039681, 0.0040239
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031318, 0.0030883
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002664, 0.0002702

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007051, upper bound: 0.0007357
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007051, upper bound: 0.0007382
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027512, 0.0027110
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007757, 0.0007643
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057231, 0.0056394
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007574, 0.0007463
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042145, 0.0042771
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011709, 0.0011883
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010628, 0.0010786
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039663, 0.0040252
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031328, 0.0030870
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002663, 0.0002703

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006920, upper bound: 0.0007060
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0007096
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029408, 0.0030008
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008291, 0.0008460
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061174, 0.0062423
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008095, 0.0008261
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046651, 0.0045717
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012961, 0.0012702
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011765, 0.0011529
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043904, 0.0043025
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033487, 0.0034170
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002948, 0.0002889

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007207, upper bound: 0.0007199
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007207, upper bound: 0.0007199
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029428, 0.0029990
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008297, 0.0008455
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061217, 0.0062385
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008101, 0.0008256
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046623, 0.0045750
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012953, 0.0012711
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011758, 0.0011537
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043877, 0.0043055
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033510, 0.0034150
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002946, 0.0002891

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006970, upper bound: 0.0006971
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007303, upper bound: 0.0006997
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029394, 0.0029848
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008287, 0.0008415
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061145, 0.0062090
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008092, 0.0008217
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046402, 0.0045696
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012892, 0.0012696
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011702, 0.0011524
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043670, 0.0043005
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033471, 0.0033988
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002932, 0.0002888

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007124
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007001
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029670, 0.0029711
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008365, 0.0008377
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061719, 0.0061805
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008168, 0.0008179
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046189, 0.0046125
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012833, 0.0012815
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011648, 0.0011632
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043469, 0.0043409
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033785, 0.0033832
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002919, 0.0002915

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006969, upper bound: 0.0007081
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006969, upper bound: 0.0007078
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029618, 0.0029369
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008350, 0.0008280
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061611, 0.0061093
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008153, 0.0008085
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045657, 0.0046044
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012685, 0.0012792
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011514, 0.0011612
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042969, 0.0043333
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033726, 0.0033442
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002885, 0.0002910

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007189
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007189
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029685, 0.0029294
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008369, 0.0008259
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061750, 0.0060938
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008172, 0.0008064
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045541, 0.0046148
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012653, 0.0012821
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011485, 0.0011638
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042859, 0.0043431
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033802, 0.0033357
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002878, 0.0002916

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006896, upper bound: 0.0006909
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006838, upper bound: 0.0006921
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029989, 0.0029430
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008455, 0.0008297
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062384, 0.0061220
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008255, 0.0008101
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045752, 0.0046622
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012711, 0.0012953
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011538, 0.0011757
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043057, 0.0043876
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034149, 0.0033512
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002891, 0.0002946

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007308
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007304
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030013, 0.0029409
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008462, 0.0008291
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062433, 0.0061176
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008262, 0.0008096
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045719, 0.0046659
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012702, 0.0012963
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011530, 0.0011767
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043027, 0.0043911
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034176, 0.0033488
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002889, 0.0002949

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007201, upper bound: 0.0007512
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007201, upper bound: 0.0007508
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029950, 0.0030309
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008444, 0.0008545
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062303, 0.0063049
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008245, 0.0008343
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047119, 0.0046561
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013091, 0.0012936
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011883, 0.0011742
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044344, 0.0043819
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034105, 0.0034513
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002978, 0.0002942

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007406, upper bound: 0.0007396
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007424, upper bound: 0.0007386
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030386, 0.0030086
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008567, 0.0008482
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0063209, 0.0062584
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008365, 0.0008282
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046772, 0.0047238
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012995, 0.0013124
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011795, 0.0011913
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044017, 0.0044457
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034601, 0.0034259
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002956, 0.0002985

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006484, upper bound: 0.0006489
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006489, upper bound: 0.0006480
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030012, 0.0030241
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008462, 0.0008526
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062432, 0.0062908
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008262, 0.0008325
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047014, 0.0046658
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013062, 0.0012963
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011856, 0.0011766
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044245, 0.0043910
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034175, 0.0034436
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002971, 0.0002948

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007687, upper bound: 0.0007682
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007687, upper bound: 0.0007686
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030152, 0.0030114
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008501, 0.0008490
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062722, 0.0062643
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008300, 0.0008290
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046815, 0.0046874
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013007, 0.0013023
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011806, 0.0011821
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044058, 0.0044114
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034334, 0.0034291
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002958, 0.0002962

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007518, upper bound: 0.0007650
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007518, upper bound: 0.0007634
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029743, 0.0030207
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008386, 0.0008516
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061872, 0.0062836
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008188, 0.0008315
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046960, 0.0046240
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013047, 0.0012847
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011843, 0.0011661
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044195, 0.0043517
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033869, 0.0034397
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002968, 0.0002922

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007356, upper bound: 0.0007196
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007356, upper bound: 0.0007195
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029883, 0.0030071
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008425, 0.0008478
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062164, 0.0062554
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008226, 0.0008278
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046749, 0.0046457
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012988, 0.0012907
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011789, 0.0011716
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043996, 0.0043721
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034029, 0.0034242
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002954, 0.0002936

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007520, upper bound: 0.0007537
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007520, upper bound: 0.0007559
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030149, 0.0029940
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008500, 0.0008441
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062716, 0.0062281
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008300, 0.0008242
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046545, 0.0046870
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012932, 0.0013022
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011738, 0.0011820
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043804, 0.0044110
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034331, 0.0034093
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002941, 0.0002962

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007215
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007198, upper bound: 0.0007167
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030228, 0.0029865
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008522, 0.0008420
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062880, 0.0062125
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008321, 0.0008221
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046429, 0.0046993
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012899, 0.0013056
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011709, 0.0011851
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043695, 0.0044225
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034421, 0.0034008
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002934, 0.0002970

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007254, upper bound: 0.0007284
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007254, upper bound: 0.0007318
time: 2.14 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007593, upper bound: 0.0007744
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007593, upper bound: 0.0007740
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007658, upper bound: 0.0007804
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007658, upper bound: 0.0007911
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007299, upper bound: 0.0007323
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007178, upper bound: 0.0007449
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007487, upper bound: 0.0007583
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007483, upper bound: 0.0007588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006545, upper bound: 0.0006646
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006545, upper bound: 0.0006646
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006874, upper bound: 0.0006990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006874, upper bound: 0.0006990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007175, upper bound: 0.0007369
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007175, upper bound: 0.0007373
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007353, upper bound: 0.0007498
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007353, upper bound: 0.0007502
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006989, upper bound: 0.0007006
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0007054
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006990, upper bound: 0.0007005
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0007054
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007051, upper bound: 0.0007357
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007051, upper bound: 0.0007382
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006920, upper bound: 0.0007060
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0007096
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007207, upper bound: 0.0007199
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007207, upper bound: 0.0007199
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006970, upper bound: 0.0006971
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007303, upper bound: 0.0006997
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007124
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007001
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006969, upper bound: 0.0007081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006969, upper bound: 0.0007078
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007189
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007189
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006896, upper bound: 0.0006909
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006838, upper bound: 0.0006921
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007308
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007304
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007201, upper bound: 0.0007512
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007201, upper bound: 0.0007508
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007406, upper bound: 0.0007396
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007424, upper bound: 0.0007386
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006484, upper bound: 0.0006489
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0006489, upper bound: 0.0006480
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007687, upper bound: 0.0007682
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007687, upper bound: 0.0007686
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007518, upper bound: 0.0007650
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007518, upper bound: 0.0007634
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007356, upper bound: 0.0007196
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007356, upper bound: 0.0007195
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007520, upper bound: 0.0007537
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007520, upper bound: 0.0007559
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007215
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007198, upper bound: 0.0007167
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007254, upper bound: 0.0007284
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.21
Output dim: 5, lower bound: -0.0007254, upper bound: 0.0007318

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029839, 0.0029571
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008413, 0.0008337
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062070, 0.0061513
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008214, 0.0008140
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045971, 0.0046387
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012772, 0.0012888
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011593, 0.0011698
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043264, 0.0043656
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033977, 0.0033672
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002905, 0.0002931

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007454, upper bound: 0.0007623
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007454, upper bound: 0.0007658
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029767, 0.0029651
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008392, 0.0008360
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061922, 0.0061681
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008194, 0.0008162
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046096, 0.0046277
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012807, 0.0012857
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011625, 0.0011670
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043382, 0.0043551
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033896, 0.0033764
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002913, 0.0002924

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007344, upper bound: 0.0007472
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007344, upper bound: 0.0007472
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029632, 0.0029543
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008354, 0.0008329
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061641, 0.0061456
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008157, 0.0008133
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045929, 0.0046066
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012760, 0.0012799
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011583, 0.0011617
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043224, 0.0043354
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033742, 0.0033641
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002902, 0.0002911

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007678
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007610
time: 2.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029756, 0.0029404
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008389, 0.0008290
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061898, 0.0061167
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008191, 0.0008094
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045712, 0.0046258
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012700, 0.0012852
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011528, 0.0011666
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043020, 0.0043534
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033883, 0.0033483
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002889, 0.0002923

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007780
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007736
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027494, 0.0027535
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007752, 0.0007763
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057193, 0.0057279
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007569, 0.0007580
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042807, 0.0042742
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011893, 0.0011875
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010795, 0.0010779
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040286, 0.0040225
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031307, 0.0031354
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002705, 0.0002701

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007272, upper bound: 0.0007286
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007240, upper bound: 0.0007303
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027864, 0.0027291
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007856, 0.0007694
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057964, 0.0056770
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007671, 0.0007513
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042426, 0.0043319
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011787, 0.0012035
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010699, 0.0010924
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039928, 0.0040768
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031729, 0.0031076
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002681, 0.0002737

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006943, upper bound: 0.0007103
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006943, upper bound: 0.0007099
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027687, 0.0027515
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007806, 0.0007757
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057594, 0.0057237
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007622, 0.0007574
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042775, 0.0043042
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011884, 0.0011958
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010787, 0.0010855
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040256, 0.0040508
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031527, 0.0031331
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002703, 0.0002720

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007194, upper bound: 0.0007196
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007139, upper bound: 0.0007255
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027846, 0.0027506
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007851, 0.0007755
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057925, 0.0057218
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007665, 0.0007572
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042761, 0.0043289
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011880, 0.0012027
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010784, 0.0010917
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040243, 0.0040740
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031708, 0.0031321
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002702, 0.0002736

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007455, upper bound: 0.0007558
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007456, upper bound: 0.0007558
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029579, 0.0029110
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008339, 0.0008207
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061531, 0.0060555
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008143, 0.0008014
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045255, 0.0045984
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012573, 0.0012776
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011413, 0.0011597
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042590, 0.0043276
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033682, 0.0033148
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002860, 0.0002906

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006318, upper bound: 0.0006355
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006318, upper bound: 0.0006355
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029657, 0.0029033
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008361, 0.0008186
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061693, 0.0060395
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008164, 0.0007992
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045136, 0.0046105
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012540, 0.0012809
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011383, 0.0011627
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042478, 0.0043390
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033771, 0.0033060
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002852, 0.0002914

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007160, upper bound: 0.0007258
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007078, upper bound: 0.0007357
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029853, 0.0029335
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008417, 0.0008271
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062101, 0.0061022
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008218, 0.0008075
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045604, 0.0046411
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012670, 0.0012894
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011501, 0.0011704
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042919, 0.0043677
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033994, 0.0033404
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002882, 0.0002933

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007187, upper bound: 0.0007316
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007187, upper bound: 0.0007316
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029862, 0.0029322
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008419, 0.0008267
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062120, 0.0060996
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008221, 0.0008072
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045585, 0.0046424
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012665, 0.0012898
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011496, 0.0011708
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042900, 0.0043691
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034004, 0.0033389
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002881, 0.0002934

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007213, upper bound: 0.0007394
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007232, upper bound: 0.0007343
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027170, 0.0026872
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007660, 0.0007576
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0056519, 0.0055899
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007479, 0.0007397
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0041776, 0.0042238
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011606, 0.0011735
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010535, 0.0010652
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039315, 0.0039751
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0030938, 0.0030599
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002640, 0.0002669

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006717, upper bound: 0.0006837
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006618, upper bound: 0.0006837
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027236, 0.0026788
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007679, 0.0007553
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0056656, 0.0055725
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007498, 0.0007374
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0041646, 0.0042341
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011570, 0.0011764
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010502, 0.0010678
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039193, 0.0039848
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031014, 0.0030504
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002632, 0.0002676

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006896, upper bound: 0.0007211
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006896, upper bound: 0.0007206
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029310, 0.0029922
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008263, 0.0008436
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060970, 0.0062243
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008068, 0.0008237
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046517, 0.0045565
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012924, 0.0012659
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011731, 0.0011491
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043777, 0.0042882
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033375, 0.0034072
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002940, 0.0002879

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007059, upper bound: 0.0006851
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006866
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029321, 0.0029912
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008267, 0.0008433
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060994, 0.0062223
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008072, 0.0008234
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046502, 0.0045583
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012920, 0.0012664
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011727, 0.0011495
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043763, 0.0042899
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033388, 0.0034061
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002939, 0.0002881

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007063, upper bound: 0.0007043
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007058
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029225, 0.0029701
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008240, 0.0008374
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060793, 0.0061783
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008045, 0.0008176
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046173, 0.0045433
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012828, 0.0012623
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011644, 0.0011458
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043454, 0.0042758
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033278, 0.0033820
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002918, 0.0002871

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007278, upper bound: 0.0006965
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006938, upper bound: 0.0006967
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0028667, 0.0029028
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008082, 0.0008184
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0059633, 0.0060385
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007891, 0.0007991
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045128, 0.0044566
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012538, 0.0012382
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011381, 0.0011239
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042470, 0.0041942
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0032643, 0.0033055
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002852, 0.0002816

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006028, upper bound: 0.0006095
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006028, upper bound: 0.0006095
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029469, 0.0029235
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008308, 0.0008242
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061302, 0.0060814
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008112, 0.0008048
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045449, 0.0045813
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012627, 0.0012728
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011461, 0.0011553
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042772, 0.0043115
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033557, 0.0033290
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002872, 0.0002895

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006991, upper bound: 0.0007157
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006991, upper bound: 0.0007154
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029618, 0.0029220
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008350, 0.0008238
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061611, 0.0060784
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008153, 0.0008044
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045426, 0.0046044
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012621, 0.0012792
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011456, 0.0011612
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042751, 0.0043333
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033726, 0.0033273
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002871, 0.0002910

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006657, upper bound: 0.0006763
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0006763
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029625, 0.0029140
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008352, 0.0008216
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061627, 0.0060618
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008155, 0.0008022
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045302, 0.0046056
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012586, 0.0012796
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011425, 0.0011615
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042634, 0.0043344
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033735, 0.0033182
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002863, 0.0002910

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006119, upper bound: 0.0006249
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006119, upper bound: 0.0006249
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029700, 0.0029070
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008374, 0.0008196
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061782, 0.0060472
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008176, 0.0008003
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045193, 0.0046172
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012556, 0.0012828
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011397, 0.0011644
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042532, 0.0043453
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033820, 0.0033103
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002856, 0.0002918

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007257
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007279
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029913, 0.0029322
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008434, 0.0008267
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062225, 0.0060996
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008235, 0.0008072
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045585, 0.0046503
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012665, 0.0012920
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011496, 0.0011727
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042900, 0.0043765
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034062, 0.0033389
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002881, 0.0002939

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007254
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007294
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029926, 0.0029317
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008437, 0.0008265
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062253, 0.0060985
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008238, 0.0008070
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045576, 0.0046524
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012662, 0.0012926
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011494, 0.0011733
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042892, 0.0043784
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034077, 0.0033383
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002880, 0.0002940

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006613, upper bound: 0.0006769
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006608, upper bound: 0.0006769
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029950, 0.0030300
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008444, 0.0008543
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062303, 0.0063029
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008245, 0.0008341
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047104, 0.0046561
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013087, 0.0012936
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011879, 0.0011742
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044330, 0.0043819
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034105, 0.0034502
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002977, 0.0002942

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0006961
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0006961
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029941, 0.0030304
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008442, 0.0008544
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062284, 0.0063038
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008242, 0.0008342
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047111, 0.0046547
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013089, 0.0012932
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011881, 0.0011738
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044336, 0.0043806
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034094, 0.0034507
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002977, 0.0002941

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006802, upper bound: 0.0006771
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006802, upper bound: 0.0006771
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030031, 0.0030267
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008467, 0.0008534
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062470, 0.0062962
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008267, 0.0008332
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047054, 0.0046686
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013073, 0.0012971
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011866, 0.0011774
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044283, 0.0043937
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034196, 0.0034466
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002974, 0.0002950

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007523, upper bound: 0.0007572
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007523, upper bound: 0.0007532
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030039, 0.0030259
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008469, 0.0008531
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062486, 0.0062945
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008269, 0.0008330
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0047041, 0.0046698
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013069, 0.0012974
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011863, 0.0011777
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044271, 0.0043948
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034205, 0.0034456
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002973, 0.0002951

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007418, upper bound: 0.0007505
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007501, upper bound: 0.0007424
time: 2.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029990, 0.0029867
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008455, 0.0008421
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062384, 0.0062130
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008256, 0.0008222
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046432, 0.0046622
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012900, 0.0012953
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011709, 0.0011757
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043698, 0.0043877
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034149, 0.0034010
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002934, 0.0002946

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006794, upper bound: 0.0006860
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006794, upper bound: 0.0006864
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029905, 0.0029938
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008431, 0.0008441
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062209, 0.0062277
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008232, 0.0008241
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046542, 0.0046491
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012931, 0.0012917
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011737, 0.0011724
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043801, 0.0043753
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034053, 0.0034090
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002941, 0.0002938

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007268, upper bound: 0.0007343
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007260, upper bound: 0.0007352
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029384, 0.0029918
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008285, 0.0008435
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061125, 0.0062236
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008089, 0.0008236
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046511, 0.0045681
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012922, 0.0012692
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011729, 0.0011520
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043772, 0.0042991
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033460, 0.0034068
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002939, 0.0002887

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0006681
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006822, upper bound: 0.0006815
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029455, 0.0029850
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008304, 0.0008416
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061272, 0.0062093
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008108, 0.0008217
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046405, 0.0045791
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012893, 0.0012722
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011703, 0.0011548
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043672, 0.0043094
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033540, 0.0033990
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002932, 0.0002894

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007193, upper bound: 0.0007081
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007238, upper bound: 0.0007023
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029901, 0.0030098
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008430, 0.0008486
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062201, 0.0062609
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008231, 0.0008285
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046790, 0.0046485
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013000, 0.0012915
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011800, 0.0011723
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044035, 0.0043748
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034049, 0.0034272
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002957, 0.0002938

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007258, upper bound: 0.0007257
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007246, upper bound: 0.0007263
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029910, 0.0030089
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008433, 0.0008483
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062219, 0.0062591
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008234, 0.0008283
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046776, 0.0046498
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012996, 0.0012919
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011796, 0.0011726
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0044022, 0.0043760
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034059, 0.0034262
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002956, 0.0002938

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007188, upper bound: 0.0007277
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007188, upper bound: 0.0007288
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027990, 0.0027769
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007891, 0.0007829
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0058224, 0.0057766
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007705, 0.0007644
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0043171, 0.0043513
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011994, 0.0012089
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010887, 0.0010973
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040628, 0.0040951
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031872, 0.0031621
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002728, 0.0002750

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007072, upper bound: 0.0007104
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007031, upper bound: 0.0007125
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027975, 0.0027772
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007887, 0.0007830
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0058193, 0.0057772
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007701, 0.0007645
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0043175, 0.0043490
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011995, 0.0012083
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010888, 0.0010967
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040633, 0.0040929
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031855, 0.0031624
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002728, 0.0002748

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006822, upper bound: 0.0006642
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006615, upper bound: 0.0006762
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030232, 0.0029873
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008524, 0.0008422
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062889, 0.0062143
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008322, 0.0008224
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046442, 0.0046999
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012903, 0.0013058
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011712, 0.0011853
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043707, 0.0044232
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034426, 0.0034017
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002935, 0.0002970

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005127, upper bound: 0.0005120
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005127, upper bound: 0.0005120
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0030241, 0.0029866
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008526, 0.0008420
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062907, 0.0062127
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008325, 0.0008222
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046430, 0.0047013
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012900, 0.0013062
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011709, 0.0011856
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043696, 0.0044244
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0034435, 0.0034008
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002934, 0.0002971

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0006779
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006732, upper bound: 0.0006897
time: 2.24 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007454, upper bound: 0.0007623
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007454, upper bound: 0.0007658
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007344, upper bound: 0.0007472
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007344, upper bound: 0.0007472
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007678
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007610
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007780
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007498, upper bound: 0.0007736
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007272, upper bound: 0.0007286
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007240, upper bound: 0.0007303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006943, upper bound: 0.0007103
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006943, upper bound: 0.0007099
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007194, upper bound: 0.0007196
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007139, upper bound: 0.0007255
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007455, upper bound: 0.0007558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007456, upper bound: 0.0007558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006318, upper bound: 0.0006355
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006318, upper bound: 0.0006355
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007160, upper bound: 0.0007258
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007078, upper bound: 0.0007357
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007187, upper bound: 0.0007316
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007187, upper bound: 0.0007316
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007213, upper bound: 0.0007394
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007232, upper bound: 0.0007343
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006717, upper bound: 0.0006837
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006618, upper bound: 0.0006837
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006896, upper bound: 0.0007211
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006896, upper bound: 0.0007206
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007059, upper bound: 0.0006851
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006857, upper bound: 0.0006866
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007063, upper bound: 0.0007043
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007058
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007278, upper bound: 0.0006965
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006938, upper bound: 0.0006967
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006028, upper bound: 0.0006095
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006028, upper bound: 0.0006095
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006991, upper bound: 0.0007157
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006991, upper bound: 0.0007154
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006657, upper bound: 0.0006763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0006763
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006119, upper bound: 0.0006249
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006119, upper bound: 0.0006249
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007257
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007279
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007254
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007294
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006613, upper bound: 0.0006769
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006608, upper bound: 0.0006769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0006961
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0006961
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006802, upper bound: 0.0006771
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006802, upper bound: 0.0006771
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007523, upper bound: 0.0007572
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007523, upper bound: 0.0007532
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007418, upper bound: 0.0007505
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007501, upper bound: 0.0007424
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006794, upper bound: 0.0006860
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006794, upper bound: 0.0006864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007268, upper bound: 0.0007343
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007260, upper bound: 0.0007352
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006738, upper bound: 0.0006681
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006822, upper bound: 0.0006815
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007193, upper bound: 0.0007081
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007238, upper bound: 0.0007023
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007258, upper bound: 0.0007257
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007246, upper bound: 0.0007263
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007188, upper bound: 0.0007277
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007188, upper bound: 0.0007288
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007072, upper bound: 0.0007104
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0007031, upper bound: 0.0007125
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006822, upper bound: 0.0006642
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006615, upper bound: 0.0006762
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0005127, upper bound: 0.0005120
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0005127, upper bound: 0.0005120
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0006779
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.02
Output dim: 5, lower bound: -0.0006732, upper bound: 0.0006897

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029456, 0.0029272
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008305, 0.0008253
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061275, 0.0060891
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008109, 0.0008058
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045506, 0.0045793
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012643, 0.0012723
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011476, 0.0011548
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042826, 0.0043096
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033542, 0.0033332
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002876, 0.0002894

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007409, upper bound: 0.0007567
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007409, upper bound: 0.0007567
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029535, 0.0029188
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008327, 0.0008229
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061440, 0.0060717
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008131, 0.0008035
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045376, 0.0045916
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012607, 0.0012757
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011443, 0.0011579
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042704, 0.0043212
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033632, 0.0033237
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002867, 0.0002902

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007362, upper bound: 0.0007560
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007362, upper bound: 0.0007649
time: 2.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029384, 0.0029347
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008285, 0.0008274
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061125, 0.0061048
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008089, 0.0008079
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045624, 0.0045681
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012676, 0.0012692
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011506, 0.0011520
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042937, 0.0042991
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033460, 0.0033418
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002883, 0.0002887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007273, upper bound: 0.0007416
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007292, upper bound: 0.0007425
time: 2.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029463, 0.0029280
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008307, 0.0008255
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061290, 0.0060909
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008111, 0.0008060
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045520, 0.0045804
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012647, 0.0012726
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011479, 0.0011551
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042839, 0.0043107
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033550, 0.0033342
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002877, 0.0002895

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007358, upper bound: 0.0007380
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007259, upper bound: 0.0007460
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027512, 0.0027408
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007757, 0.0007727
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057231, 0.0057014
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007574, 0.0007545
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042608, 0.0042771
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011838, 0.0011883
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010745, 0.0010786
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040099, 0.0040252
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031328, 0.0031209
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002693, 0.0002703

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007206
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007403
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027496, 0.0027407
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007752, 0.0007727
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057198, 0.0057012
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007569, 0.0007545
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042607, 0.0042746
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011838, 0.0011876
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010745, 0.0010780
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040098, 0.0040229
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031310, 0.0031209
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002693, 0.0002701

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007256, upper bound: 0.0007376
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007367, upper bound: 0.0007369
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027640, 0.0027269
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007793, 0.0007688
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057497, 0.0056724
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007609, 0.0007507
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042392, 0.0042970
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011778, 0.0011938
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010691, 0.0010836
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039896, 0.0040439
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031474, 0.0031051
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002679, 0.0002715

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007278
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007517
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027620, 0.0027262
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007787, 0.0007686
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057455, 0.0056710
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007603, 0.0007505
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042382, 0.0042938
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011775, 0.0011930
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010688, 0.0010828
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039886, 0.0040410
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031451, 0.0031043
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002678, 0.0002713

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007222
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007466
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027499, 0.0027549
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007753, 0.0007767
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057205, 0.0057308
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007570, 0.0007584
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042828, 0.0042751
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011899, 0.0011878
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010801, 0.0010781
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040306, 0.0040234
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031314, 0.0031370
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002706, 0.0002702

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006924, upper bound: 0.0006982
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006924, upper bound: 0.0006979
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027508, 0.0027537
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007756, 0.0007764
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057222, 0.0057283
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007572, 0.0007580
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042810, 0.0042764
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011894, 0.0011881
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010796, 0.0010785
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040289, 0.0040246
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031323, 0.0031357
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002705, 0.0002702

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007253
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007253
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027263, 0.0027344
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007687, 0.0007709
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0056713, 0.0056881
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007505, 0.0007527
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042509, 0.0042384
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011810, 0.0011776
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010720, 0.0010689
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040006, 0.0039888
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031045, 0.0031137
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002686, 0.0002678

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006666, upper bound: 0.0006556
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006541, upper bound: 0.0006686
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027687, 0.0027091
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007806, 0.0007638
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057594, 0.0056356
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007622, 0.0007458
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042117, 0.0043042
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011701, 0.0011958
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010621, 0.0010855
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039637, 0.0040508
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031527, 0.0030849
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002662, 0.0002720

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006203, upper bound: 0.0006231
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006203, upper bound: 0.0006231
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027749, 0.0027418
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007824, 0.0007730
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057724, 0.0057035
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007639, 0.0007548
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042624, 0.0043139
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011842, 0.0011985
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010749, 0.0010879
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040114, 0.0040599
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031598, 0.0031221
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002694, 0.0002726

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0007120
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007039, upper bound: 0.0007159
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027758, 0.0027408
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007826, 0.0007727
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057741, 0.0057013
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007641, 0.0007545
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042608, 0.0043152
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011838, 0.0011989
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010745, 0.0010882
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0040099, 0.0040611
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031608, 0.0031209
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002693, 0.0002727

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007241, upper bound: 0.0007324
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007244, upper bound: 0.0007324
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029253, 0.0028749
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008248, 0.0008106
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060853, 0.0059804
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008053, 0.0007914
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0044694, 0.0045477
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012417, 0.0012635
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011271, 0.0011469
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042062, 0.0042799
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033311, 0.0032737
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002824, 0.0002874

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006791, upper bound: 0.0006916
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006791, upper bound: 0.0006916
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029381, 0.0028621
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008284, 0.0008069
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061118, 0.0059538
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008088, 0.0007879
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0044495, 0.0045676
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012362, 0.0012690
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011221, 0.0011519
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0041874, 0.0042986
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033456, 0.0032591
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002812, 0.0002886

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006628, upper bound: 0.0006829
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006628, upper bound: 0.0006825
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029701, 0.0029196
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008374, 0.0008232
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061784, 0.0060734
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008176, 0.0008037
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045389, 0.0046173
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012610, 0.0012828
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011446, 0.0011644
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042716, 0.0043454
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033820, 0.0033246
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002868, 0.0002918

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007060, upper bound: 0.0007190
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007060, upper bound: 0.0007228
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029853, 0.0029186
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008417, 0.0008229
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0062101, 0.0060713
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008218, 0.0008034
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045373, 0.0046411
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012606, 0.0012894
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011442, 0.0011704
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042701, 0.0043677
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033994, 0.0033234
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002867, 0.0002933

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006610, upper bound: 0.0006673
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006610, upper bound: 0.0006673
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027743, 0.0027184
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007822, 0.0007664
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057711, 0.0056548
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007637, 0.0007483
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042261, 0.0043129
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011741, 0.0011983
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010658, 0.0010877
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039772, 0.0040590
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031591, 0.0030955
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002671, 0.0002726

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006963
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006963
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027708, 0.0027174
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007812, 0.0007661
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0057639, 0.0056528
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007628, 0.0007481
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0042245, 0.0043076
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011737, 0.0011968
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010654, 0.0010863
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0039757, 0.0040539
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0031552, 0.0030943
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002670, 0.0002722

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007294
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007316
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0027066, 0.0026543
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007631, 0.0007483
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0056303, 0.0055214
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007451, 0.0007307
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0041264, 0.0042078
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011464, 0.0011690
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010406, 0.0010611
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0038834, 0.0039600
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0030821, 0.0030224
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002608, 0.0002659

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006812, upper bound: 0.0007111
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006812, upper bound: 0.0007198
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0026990, 0.0026637
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0007610, 0.0007510
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0056145, 0.0055410
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0007430, 0.0007333
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0041410, 0.0041960
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0011505, 0.0011658
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0010443, 0.0010582
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0038971, 0.0039489
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0030734, 0.0030331
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002617, 0.0002652

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006044, upper bound: 0.0006087
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006044, upper bound: 0.0006087
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029245, 0.0029727
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008245, 0.0008381
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0060836, 0.0061838
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008051, 0.0008183
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0046214, 0.0045465
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012840, 0.0012631
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011654, 0.0011466
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0043492, 0.0042787
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033302, 0.0033850
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002920, 0.0002873

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004843, upper bound: 0.0004799
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004843, upper bound: 0.0004799
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029487, 0.0029238
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008314, 0.0008243
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061340, 0.0060820
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008117, 0.0008049
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045453, 0.0045841
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012628, 0.0012736
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011463, 0.0011561
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042776, 0.0043142
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033577, 0.0033293
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002872, 0.0002897

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006611, upper bound: 0.0006735
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006714, upper bound: 0.0006739
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029472, 0.0029245
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008309, 0.0008245
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061308, 0.0060836
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008113, 0.0008051
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045465, 0.0045818
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012632, 0.0012730
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011466, 0.0011555
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042788, 0.0043120
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033560, 0.0033302
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002873, 0.0002895

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006720, upper bound: 0.0006814
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006715, upper bound: 0.0006821
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0029722, 0.0029100
1: -0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008380, 0.0008204
2: 0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0061827, 0.0060533
3: 0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008182, 0.0008011
4: -0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0045239, 0.0046206
5: 0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0012569, 0.0012837
6: 0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0011409, 0.0011652
7: -0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0042575, 0.0043485
8: -0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0033844, 0.0033136
9: -0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0002859, 0.0002920

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006690, upper bound: 0.0007072
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006690, upper bound: 0.0006969
time: 2.18 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 5.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007409, upper bound: 0.0007567
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007409, upper bound: 0.0007567
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007362, upper bound: 0.0007560
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007362, upper bound: 0.0007649
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007273, upper bound: 0.0007416
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007292, upper bound: 0.0007425
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007358, upper bound: 0.0007380
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007259, upper bound: 0.0007460
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007206
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007403
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007256, upper bound: 0.0007376
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007367, upper bound: 0.0007369
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007278
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007517
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007222
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007014, upper bound: 0.0007466
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006924, upper bound: 0.0006982
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006924, upper bound: 0.0006979
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006666, upper bound: 0.0006556
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006541, upper bound: 0.0006686
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006203, upper bound: 0.0006231
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006203, upper bound: 0.0006231
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0007120
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007039, upper bound: 0.0007159
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007241, upper bound: 0.0007324
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007244, upper bound: 0.0007324
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006791, upper bound: 0.0006916
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006791, upper bound: 0.0006916
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006628, upper bound: 0.0006829
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006628, upper bound: 0.0006825
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007060, upper bound: 0.0007190
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007060, upper bound: 0.0007228
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006610, upper bound: 0.0006673
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006610, upper bound: 0.0006673
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006963
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006963
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007294
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007316
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006812, upper bound: 0.0007111
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006812, upper bound: 0.0007198
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006044, upper bound: 0.0006087
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006044, upper bound: 0.0006087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0004843, upper bound: 0.0004799
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0004843, upper bound: 0.0004799
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006611, upper bound: 0.0006735
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006714, upper bound: 0.0006739
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006720, upper bound: 0.0006814
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006715, upper bound: 0.0006821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006690, upper bound: 0.0007072
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.31
Output dim: 5, lower bound: -0.0006690, upper bound: 0.0006969
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007279
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007254
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007046, upper bound: 0.0007294
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007523, upper bound: 0.0007572
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007523, upper bound: 0.0007532
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007418, upper bound: 0.0007505
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007501, upper bound: 0.0007424
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007268, upper bound: 0.0007343
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007260, upper bound: 0.0007352
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007193, upper bound: 0.0007081
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007238, upper bound: 0.0007023
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007258, upper bound: 0.0007257
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007246, upper bound: 0.0007263
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007188, upper bound: 0.0007277
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007188, upper bound: 0.0007288
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.31
Output dim: 5, lower bound: -0.0007031, upper bound: 0.0007125

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.95 + 597.71 = 601.66 seconds
