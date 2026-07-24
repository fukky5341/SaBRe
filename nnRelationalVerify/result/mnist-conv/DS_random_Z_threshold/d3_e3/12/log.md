## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.9345672972


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8727684, 1.8727684)
1: (-21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0088158, 2.0088162)
2: (-5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7741785, 1.7741785)
3: (-13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6917539, 1.6917539)
4: (-8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649471, 1.5649471)
5: (-7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4956284, 1.4956284)
6: (-5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4957285, 1.4957283)
7: (-10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3836555, 2.3836555)
8: (-3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7785492, 1.7785492)
9: (-4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6223621, 1.6223621)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.61 + 34.24 = 55.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9355028, upper bound: 0.9355027

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 500

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354960, upper bound: 0.9262078
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262081, upper bound: 0.9354959
time: 4.28 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.89 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 0, lower bound: -0.9354960, upper bound: 0.9262078
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 0, lower bound: -0.9262081, upper bound: 0.9354959

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8714519, 1.8689475
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0074477, 2.0048423
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7735300, 1.7722945
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6905036, 1.6913218
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649123, 1.5648465
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4937019, 1.4949641
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4933572, 1.4949112
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3834000, 2.3829145
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7777376, 1.7762017
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6216021, 1.6220999

Time for backsubstitution: 19.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5778

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9319327, upper bound: 0.9262043
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354913, upper bound: 0.9226444
time: 4.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8689475, 1.8714528
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0048432, 2.0074477
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7722940, 1.7735295
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6913218, 1.6905041
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5648465, 1.5649123
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4949641, 1.4937019
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4949112, 1.4933572
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3829145, 2.3834000
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7762012, 1.7777371
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6220999, 1.6216021

Time for backsubstitution: 20.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9262080, upper bound: 0.9233075
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9261156, upper bound: 0.9354942
time: 4.47 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.34 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 29.34
Output dim: 0, lower bound: -0.9319327, upper bound: 0.9262043
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.34
Output dim: 0, lower bound: -0.9354913, upper bound: 0.9226444
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 29.34
Output dim: 0, lower bound: -0.9262080, upper bound: 0.9233075
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.34
Output dim: 0, lower bound: -0.9261156, upper bound: 0.9354942

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8681350, 1.8635836
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0048280, 2.0032158
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7734432, 1.7721491
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6891232, 1.6890869
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5595317, 1.5615187
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4935417, 1.4947042
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4930711, 1.4944479
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3796167, 2.3805761
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7724218, 1.7729149
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6197033, 1.6190372

Time for backsubstitution: 20.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354897, upper bound: 0.9225301
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9233029, upper bound: 0.9226443
time: 4.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8689456, 1.8714471
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0048385, 2.0074472
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7722921, 1.7735295
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6913209, 1.6905012
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5648365, 1.5649099
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4949622, 1.4936934
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4949059, 1.4933541
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3829126, 2.3833895
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7762012, 1.7777371
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6220989, 1.6216002

Time for backsubstitution: 20.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5844

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9261126, upper bound: 0.9348898
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255111, upper bound: 0.9354920
time: 3.93 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.88 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 0, lower bound: -0.9354897, upper bound: 0.9225301
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.88
Output dim: 0, lower bound: -0.9233029, upper bound: 0.9226443
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 0, lower bound: -0.9261126, upper bound: 0.9348898
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 0, lower bound: -0.9255111, upper bound: 0.9354920

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8681293, 1.8635817
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0048285, 2.0032115
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7734432, 1.7721472
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6891208, 1.6890864
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5595298, 1.5615091
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4935327, 1.4947023
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4930682, 1.4944427
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3796072, 2.3805742
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7724218, 1.7729149
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6197014, 1.6190357

Time for backsubstitution: 20.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 907

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9343251, upper bound: 0.9224901
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354497, upper bound: 0.9213656
time: 4.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8767595, 1.8738956
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -1.9992428, 2.0028472
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7754421, 1.7736754
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6671600, 1.6615057
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5609221, 1.5649934
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.5035539, 1.4986472
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4873524, 1.4870501
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3679323, 2.3654213
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7718401, 1.7711458
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6149507, 1.6172366

Time for backsubstitution: 20.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9226151, upper bound: 0.9319950
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9226150, upper bound: 0.9319951
time: 4.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8713942, 1.8792610
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0002384, 2.0018506
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7724390, 1.7766790
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6623259, 1.6663399
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649199, 1.5609961
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4999161, 1.5022850
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4886017, 1.4858005
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3649445, 2.3684092
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7696095, 1.7733760
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6177354, 1.6144514

Time for backsubstitution: 20.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 466

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253488, upper bound: 0.9354854
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255052, upper bound: 0.9353303
time: 4.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.59 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.59
Output dim: 0, lower bound: -0.9343251, upper bound: 0.9224901
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.59
Output dim: 0, lower bound: -0.9354497, upper bound: 0.9213656
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.59
Output dim: 0, lower bound: -0.9226151, upper bound: 0.9319950
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.59
Output dim: 0, lower bound: -0.9226150, upper bound: 0.9319951
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.59
Output dim: 0, lower bound: -0.9253488, upper bound: 0.9354854
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.59
Output dim: 0, lower bound: -0.9255052, upper bound: 0.9353303

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8664336, 1.8613539
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0034304, 2.0020714
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7736320, 1.7723889
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6890488, 1.6882730
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5549555, 1.5578461
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4937611, 1.4948802
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4916844, 1.4934235
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3776798, 2.3783255
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7700300, 1.7699833
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6187491, 1.6179037

Time for backsubstitution: 20.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9323485, upper bound: 0.9182685
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9323485, upper bound: 0.9182684
time: 4.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8710556, 1.8804345
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0001526, 2.0021462
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7725258, 1.7766523
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6617994, 1.6681595
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5646982, 1.5617704
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.5001411, 1.5022216
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4891014, 1.4856558
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3665829, 2.3679352
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7694302, 1.7739944
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6196699, 1.6138916

Time for backsubstitution: 20.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5778

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217633, upper bound: 0.9354805
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9253440, upper bound: 0.9319218
time: 5.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8713942, 1.8789215
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0002384, 2.0017648
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7724123, 1.7766790
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6623259, 1.6658139
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649199, 1.5607743
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4998527, 1.5022850
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4884572, 1.4858005
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3644705, 2.3684092
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7696095, 1.7731972
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6171751, 1.6144514

Time for backsubstitution: 20.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9226102, upper bound: 0.9318305
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9226102, upper bound: 0.9318330
time: 5.00 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.27 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.27
Output dim: 0, lower bound: -0.9323485, upper bound: 0.9182685
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.27
Output dim: 0, lower bound: -0.9323485, upper bound: 0.9182684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.27
Output dim: 0, lower bound: -0.9217633, upper bound: 0.9354805
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.27
Output dim: 0, lower bound: -0.9253440, upper bound: 0.9319218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.27
Output dim: 0, lower bound: -0.9226102, upper bound: 0.9318305
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.27
Output dim: 0, lower bound: -0.9226102, upper bound: 0.9318330

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8656902, 1.8771157
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -1.9985266, 1.9995279
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7723789, 1.7765656
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6595654, 1.6667795
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5613708, 1.5563898
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4998808, 1.5020604
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4886384, 1.4853699
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3642454, 2.3641524
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7661457, 1.7686796
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6166067, 1.6119933

Time for backsubstitution: 20.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4569

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9204269, upper bound: 0.9354374
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9217198, upper bound: 0.9341442
time: 5.17 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 30.69 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 30.69
Output dim: 0, lower bound: -0.9204269, upper bound: 0.9354374
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 30.69
Output dim: 0, lower bound: -0.9217198, upper bound: 0.9341442

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8639393, 1.8804469
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -1.9967804, 2.0028358
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7792172, 1.7729840
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6611609, 1.6659470
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5617633, 1.5561824
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4985442, 1.5046272
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4852824, 1.4917469
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3693428, 2.3614836
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7661610, 1.7686696
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6157427, 1.6136351

Time for backsubstitution: 20.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 907

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9192626, upper bound: 0.9353982
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9203874, upper bound: 0.9342737
time: 4.76 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 29.91 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 29.91
Output dim: 0, lower bound: -0.9192626, upper bound: 0.9353982
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 29.91
Output dim: 0, lower bound: -0.9203874, upper bound: 0.9342737

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8617110, 1.8787527
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -1.9956403, 2.0014386
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7794595, 1.7731733
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6603470, 1.6658745
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5581002, 1.5516076
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4987226, 1.5048561
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4842629, 1.4903626
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3670931, 2.3595562
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7632275, 1.7662764
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6146116, 1.6126828

Time for backsubstitution: 20.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9186972, upper bound: 0.9352364
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9186939, upper bound: 0.9312794
time: 6.39 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 31.94 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 31.94
Output dim: 0, lower bound: -0.9186972, upper bound: 0.9352364
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 31.94
Output dim: 0, lower bound: -0.9186939, upper bound: 0.9312794

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8611979, 1.8787699
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -1.9956565, 2.0009336
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7794642, 1.7728758
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6603384, 1.6658406
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5581183, 1.5510626
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4987431, 1.5046706
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4842277, 1.4903717
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3671093, 2.3590779
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7632475, 1.7656789
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6143007, 1.6126928

Time for backsubstitution: 20.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9157980, upper bound: 0.9317209
time: 5.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9157980, upper bound: 0.9317216
time: 4.57 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 31.19 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 31.19
Output dim: 0, lower bound: -0.9157980, upper bound: 0.9317209
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 31.19
Output dim: 0, lower bound: -0.9157980, upper bound: 0.9317216

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.85 + 428.31 = 484.15 seconds
