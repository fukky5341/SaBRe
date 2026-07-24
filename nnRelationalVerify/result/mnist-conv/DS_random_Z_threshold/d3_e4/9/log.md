## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.030627584


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9971504, 2.9971502)
1: (-7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4096041, 2.4096045)
2: (-7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3071775, 2.3071778)
3: (-11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6513076, 2.6513076)
4: (6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6619794, 1.6619792)
5: (-8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2816925, 2.2816925)
6: (-12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1651812, 3.1651816)
7: (-3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3866134, 2.3866131)
8: (-6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3960896, 2.3960896)
9: (-5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0071397, 2.0071399)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.62 + 35.49 = 58.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.0516605, upper bound: 1.0516610

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0422055, upper bound: 1.0516545
time: 5.49 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516538, upper bound: 1.0422083
time: 5.38 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.89 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 4, lower bound: -1.0422055, upper bound: 1.0516545
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 4, lower bound: -1.0516538, upper bound: 1.0422083

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9959126, 2.9913301
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4092264, 2.4095237
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3049054, 2.2965858
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6505103, 2.6476309
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6585119, 1.6612306
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2801266, 2.2743931
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1637402, 3.1648722
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3818431, 2.3855937
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3941355, 2.3870356
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0037060, 2.0063972

Time for backsubstitution: 22.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 513

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0421879, upper bound: 1.0497423
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0402882, upper bound: 1.0516368
time: 6.83 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9913311, 2.9959121
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4095240, 2.4092255
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2965856, 2.3049057
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6476312, 2.6505103
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6612308, 1.6585114
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2743931, 2.2801266
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1648722, 3.1637397
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3855939, 2.3818431
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3870354, 2.3941355
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0063972, 2.0037060

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 513

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516362, upper bound: 1.0402886
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0497395, upper bound: 1.0421883
time: 8.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 37.73 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 37.73
Output dim: 4, lower bound: -1.0421879, upper bound: 1.0497423
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 37.73
Output dim: 4, lower bound: -1.0402882, upper bound: 1.0516368
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 37.73
Output dim: 4, lower bound: -1.0516362, upper bound: 1.0402886
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 37.73
Output dim: 4, lower bound: -1.0497395, upper bound: 1.0421883

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9959860, 2.9910140
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4068146, 2.4100778
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3049097, 2.2965612
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6499820, 2.6477525
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6587520, 1.6601832
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2785997, 2.2747412
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1621084, 3.1652446
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3812575, 2.3857462
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3881674, 2.3884091
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0041215, 2.0045671

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0421864, upper bound: 1.0468478
time: 6.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0392948, upper bound: 1.0497388
time: 6.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9955959, 2.9913301
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4092264, 2.4071133
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3048811, 2.2965858
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6505103, 2.6471021
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6574640, 1.6612306
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2801266, 2.2728667
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1637402, 3.1632404
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3818431, 2.3850081
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3941355, 2.3810678
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0018761, 2.0063972

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0402867, upper bound: 1.0487427
time: 6.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0373950, upper bound: 1.0516353
time: 7.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9914036, 2.9955957
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4071131, 2.4097795
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2965899, 2.3048811
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6471019, 2.6506319
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6614709, 1.6574640
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2728672, 2.2804747
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1632404, 3.1641121
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3850083, 2.3819957
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3810682, 2.3955090
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0068123, 2.0018759

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516347, upper bound: 1.0373955
time: 7.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0487417, upper bound: 1.0402870
time: 6.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9910135, 2.9959121
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4095240, 2.4068151
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2965612, 2.3049057
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6476312, 2.6499815
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6601834, 1.6585114
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2743931, 2.2786002
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1648722, 3.1621079
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3855939, 2.3812575
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3870354, 2.3881676
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0045674, 2.0037060

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0497380, upper bound: 1.0392977
time: 7.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0468450, upper bound: 1.0421873
time: 6.80 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 36.67 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0421864, upper bound: 1.0468478
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0392948, upper bound: 1.0497388
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0402867, upper bound: 1.0487427
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0373950, upper bound: 1.0516353
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0516347, upper bound: 1.0373955
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0487417, upper bound: 1.0402870
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0497380, upper bound: 1.0392977
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.67
Output dim: 4, lower bound: -1.0468450, upper bound: 1.0421873

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9963999, 2.9913981
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4107141, 2.4134834
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3048029, 2.2964411
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6429996, 2.6416419
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6620934, 1.6626630
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2770295, 2.2733674
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1687965, 3.1729002
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3814702, 2.3859897
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3922420, 2.3930738
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0056500, 2.0063174

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2349
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 416

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2349

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0389377, upper bound: 1.0424232
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0382207, upper bound: 1.0424230
time: 5.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9963694, 2.9914281
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4102211, 2.4139767
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3047895, 2.2964547
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6438704, 2.6407707
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6612322, 1.6635246
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2772260, 2.2731705
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1697636, 3.1719327
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3815007, 2.3859589
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3928323, 2.3924835
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0058713, 2.0060961

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2349
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2852

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0376292, upper bound: 1.0495331
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0390870, upper bound: 1.0480768
time: 5.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9960098, 2.9917142
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4131250, 2.4105189
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3047743, 2.2964659
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6435289, 2.6409914
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6608059, 1.6637101
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2785563, 2.2714930
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1704273, 3.1708956
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3820553, 2.3852515
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3982100, 2.3857322
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0034046, 2.0081480

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2349
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 2608

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 424

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0365279, upper bound: 1.0474400
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0389320, upper bound: 1.0447773
time: 7.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9959803, 2.9917443
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4126320, 2.4110122
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3047609, 2.2964795
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6444006, 2.6401203
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6599438, 1.6645718
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2787528, 2.2712965
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1713943, 3.1699286
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3820863, 2.3852208
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3988004, 2.3851421
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0036259, 2.0079267

Time for backsubstitution: 20.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2216

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0370341, upper bound: 1.0515182
time: 10.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0372781, upper bound: 1.0512739
time: 8.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9918184, 2.9959798
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4110126, 2.4131851
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2964830, 2.3047609
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6401205, 2.6445212
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6648128, 1.6599436
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2712960, 2.2791004
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1699286, 3.1717677
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3852210, 2.3822391
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3851418, 2.4001734
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0083413, 2.0036261

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2349
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 3105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1858

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0488261, upper bound: 1.0344171
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0487077, upper bound: 1.0345308
time: 8.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9917879, 2.9960101
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4105186, 2.4136786
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2964697, 2.3047745
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6409922, 2.6436501
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6639512, 1.6608057
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2714925, 2.2789040
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1708956, 3.1708002
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3852515, 2.3822083
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3857322, 2.3995833
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0085626, 2.0034049

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2349
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2321

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 403

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0465455, upper bound: 1.0372704
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0457146, upper bound: 1.0381010
time: 9.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9914284, 2.9962959
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4134235, 2.4102206
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2964544, 2.3047857
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6406488, 2.6438711
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6635249, 1.6609907
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2728229, 2.2772264
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1715593, 3.1697636
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3858061, 2.3815010
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3911099, 2.3928320
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0060959, 2.0054567

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 2349
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 766

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2922

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0485253, upper bound: 1.0373993
time: 7.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0478435, upper bound: 1.0380829
time: 6.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9913988, 2.9963262
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4129305, 2.4107141
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.2964411, 2.3047993
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6415205, 2.6429996
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6626627, 1.6618528
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2730193, 2.2770300
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1725264, 3.1687961
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3858366, 2.3814702
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3917003, 2.3922420
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0063171, 2.0052354

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 414
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2378

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0456649, upper bound: 1.0416365
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0461983, upper bound: 1.0407941
time: 8.11 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 37.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0389377, upper bound: 1.0424232
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0382207, upper bound: 1.0424230
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0376292, upper bound: 1.0495331
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0390870, upper bound: 1.0480768
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0365279, upper bound: 1.0474400
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0389320, upper bound: 1.0447773
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0370341, upper bound: 1.0515182
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0372781, upper bound: 1.0512739
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0488261, upper bound: 1.0344171
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0487077, upper bound: 1.0345308
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0465455, upper bound: 1.0372704
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0457146, upper bound: 1.0381010
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0485253, upper bound: 1.0373993
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0478435, upper bound: 1.0380829
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0456649, upper bound: 1.0416365
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.50
Output dim: 4, lower bound: -1.0461983, upper bound: 1.0407941

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9958639, 2.9871786
1: -7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4103789, 2.4130032
2: -7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3053188, 2.2963223
3: -11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6334286, 2.6307073
4: 6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6573265, 1.6588464
5: -8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2749710, 2.2657290
6: -12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1671286, 3.1682515
7: -3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3805895, 2.3891547
8: -6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3901725, 2.3925331
9: -5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0057850, 2.0060427

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1802
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1404
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1244
type: DSZ, layer: 3, pos: 2852
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 572
type: DSZ, layer: 3, pos: 1982
type: DSZ, layer: 3, pos: 1247
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 894
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 416
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1486
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2244
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1778
type: DSZ, layer: 3, pos: 2336
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 3112
type: DSZ, layer: 3, pos: 176
type: DSZ, layer: 3, pos: 676
type: DSZ, layer: 3, pos: 2528
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2237
type: DSZ, layer: 3, pos: 397
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 2390
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 2922
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 760
type: DSZ, layer: 3, pos: 2608
type: DSZ, layer: 3, pos: 1858
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 2328
type: DSZ, layer: 3, pos: 1253
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1449
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 759
type: DSZ, layer: 3, pos: 1933
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 414

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1802

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0384724, upper bound: 1.0400443
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0369240, upper bound: 1.0419499
time: 5.02 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.13
Output dim: 4, lower bound: -1.0384724, upper bound: 1.0400443
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.13
Output dim: 4, lower bound: -1.0369240, upper bound: 1.0419499
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0382207, upper bound: 1.0424230
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0376292, upper bound: 1.0495331
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0390870, upper bound: 1.0480768
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0365279, upper bound: 1.0474400
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0389320, upper bound: 1.0447773
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0370341, upper bound: 1.0515182
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0372781, upper bound: 1.0512739
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0488261, upper bound: 1.0344171
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0487077, upper bound: 1.0345308
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0465455, upper bound: 1.0372704
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0457146, upper bound: 1.0381010
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0485253, upper bound: 1.0373993
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0478435, upper bound: 1.0380829
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0456649, upper bound: 1.0416365
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.13
Output dim: 4, lower bound: -1.0461983, upper bound: 1.0407941

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.12 + 543.81 = 601.92 seconds
