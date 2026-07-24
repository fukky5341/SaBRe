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
execution time: IAR + RelationalAnalysis = 24.55 + 35.56 = 60.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9355028, upper bound: 0.9355027

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5844
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 5844

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354998, upper bound: 0.9348979
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9348983, upper bound: 0.9355003
time: 5.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.00 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.00
Output dim: 0, lower bound: -0.9354998, upper bound: 0.9348979
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.00
Output dim: 0, lower bound: -0.9348983, upper bound: 0.9355003

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8805814, 1.8752151
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0032196, 2.0042157
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7773280, 1.7743239
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6675930, 1.6627588
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5610323, 1.5650296
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.5042200, 1.5005822
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4881749, 1.4894247
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3686771, 2.3656893
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7741880, 1.7719579
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6152129, 1.6179981

Time for backsubstitution: 23.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 500

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354930, upper bound: 0.9256035
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262051, upper bound: 0.9348912
time: 4.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8752160, 1.8805804
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0042152, 2.0032187
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7743249, 1.7773275
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6627584, 1.6675930
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5650301, 1.5610323
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.5005822, 1.5042200
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4894247, 1.4881749
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3656893, 2.3686771
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7719584, 1.7741885
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6179986, 1.6152134

Time for backsubstitution: 23.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 500

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9348915, upper bound: 0.9262056
time: 5.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256036, upper bound: 0.9354935
time: 5.08 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 34.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.9354930, upper bound: 0.9256035
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.9262051, upper bound: 0.9348912
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.9348915, upper bound: 0.9262056
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 0, lower bound: -0.9256036, upper bound: 0.9354935

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8792648, 1.8713946
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0018520, 2.0002437
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7766786, 1.7724404
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6663423, 1.6623259
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5609980, 1.5649300
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.5022936, 1.4999175
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4858031, 1.4886065
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3684206, 2.3649473
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7733765, 1.7696104
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6144528, 1.6177363

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9319956, upper bound: 0.9227084
time: 5.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9319949, upper bound: 0.9227087
time: 5.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8767605, 1.8738999
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -1.9992476, 2.0028486
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7754436, 1.7736754
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6671605, 1.6615081
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5609322, 1.5649958
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.5035558, 1.4986558
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4873571, 1.4870527
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3679352, 2.3654327
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7718410, 1.7711463
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6149516, 1.6172385

Time for backsubstitution: 23.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9227075, upper bound: 0.9319964
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9227072, upper bound: 0.9319960
time: 4.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8738995, 1.8767600
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0028486, 1.9992466
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7736754, 1.7754440
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6615081, 1.6671605
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649958, 1.5609322
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4986558, 1.5035558
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4870529, 1.4873569
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3654327, 2.3679352
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7711458, 1.7718406
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6172385, 1.6149516

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9319963, upper bound: 0.9227073
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9319963, upper bound: 0.9227072
time: 5.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8713951, 1.8792653
1: -21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0002432, 2.0018516
2: -5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7724404, 1.7766786
3: -13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6623259, 1.6663427
4: -8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649300, 1.5609980
5: -7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4999180, 1.5022936
6: -5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4886065, 1.4858031
7: -10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3649473, 2.3684206
8: -3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7696104, 1.7733765
9: -4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6177363, 1.6144533

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5778
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 930

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9227084, upper bound: 0.9319944
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9227084, upper bound: 0.9319950
time: 4.82 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.21 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9319956, upper bound: 0.9227084
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9319949, upper bound: 0.9227087
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9227075, upper bound: 0.9319964
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9227072, upper bound: 0.9319960
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9319963, upper bound: 0.9227073
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9319963, upper bound: 0.9227072
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9227084, upper bound: 0.9319944
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.21
Output dim: 0, lower bound: -0.9227084, upper bound: 0.9319950

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 60.11 + 210.88 = 270.99 seconds
