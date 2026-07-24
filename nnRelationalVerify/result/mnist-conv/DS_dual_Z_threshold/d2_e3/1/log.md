## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.190247616


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5856323, 0.5856323)
1: (-12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6986628, 0.6986628)
2: (-7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5276794, 0.5276794)
3: (-11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4937754, 0.4937754)
4: (-2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4948776, 0.4948779)
5: (-5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5168405, 0.5168405)
6: (7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3902711, 0.3902711)
7: (-17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6233628, 0.6233625)
8: (-3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4710871, 0.4710871)
9: (-10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7942631, 0.7942631)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.47 + 33.29 = 55.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1981739, upper bound: 0.1981734

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 4610

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1979931, upper bound: 0.1981710
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981715, upper bound: 0.1979926
time: 3.11 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.48 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.48
Output dim: 6, lower bound: -0.1979931, upper bound: 0.1981710
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.48
Output dim: 6, lower bound: -0.1981715, upper bound: 0.1979926

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5854568, 0.5859534
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6996951, 0.6980977
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5266709, 0.5295260
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4940784, 0.4936091
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4967799, 0.4938399
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5178834, 0.5162704
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3901302, 0.3905288
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6227007, 0.6245768
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4701437, 0.4728144
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7941582, 0.7944555

Time for backsubstitution: 20.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1979873, upper bound: 0.1963923
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1981666
time: 3.36 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5856323, 0.5854567
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6980977, 0.6986628
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5276794, 0.5266709
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4936092, 0.4937754
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4938397, 0.4948779
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5162708, 0.5168405
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3902711, 0.3901302
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6233628, 0.6227007
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4710871, 0.4701437
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7942631, 0.7941585

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981657, upper bound: 0.1962122
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1979883
time: 3.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.91
Output dim: 6, lower bound: -0.1979873, upper bound: 0.1963923
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.91
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1981666
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.91
Output dim: 6, lower bound: -0.1981657, upper bound: 0.1962122
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.91
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1979883

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5721583, 0.5699923
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6897011, 0.6897593
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5215945, 0.5234340
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4950666, 0.4941903
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4921303, 0.4899616
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5140378, 0.5136496
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3825815, 0.3808955
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5945089, 0.6010773
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4627128, 0.4674579
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7934747, 0.7953942

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1917875, upper bound: 0.1901748
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1917702, upper bound: 0.1901916
time: 3.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5694957, 0.5726550
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6913567, 0.6881037
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5205789, 0.5244495
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4946594, 0.4945976
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4929018, 0.4891899
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5152624, 0.5124253
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3804967, 0.3829803
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5992010, 0.5963850
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4647870, 0.4653836
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7950969, 0.7937720

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.55 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1900105, upper bound: 0.1919525
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1899924, upper bound: 0.1919698
time: 3.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5723343, 0.5694957
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6881037, 0.6903238
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5226028, 0.5205790
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4945974, 0.4943562
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4891899, 0.4910001
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5124252, 0.5142194
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3827226, 0.3804967
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5951710, 0.5992010
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4636562, 0.4647871
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7935801, 0.7950966

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.47 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1919689, upper bound: 0.1899933
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1919516, upper bound: 0.1900105
time: 3.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5696714, 0.5721583
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6897593, 0.6886683
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5215871, 0.5215944
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4941902, 0.4947634
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4899616, 0.4902283
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5136497, 0.5129951
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3806379, 0.3825816
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5998633, 0.5945086
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4657304, 0.4627128
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7952018, 0.7934749

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1901920, upper bound: 0.1917710
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1901739, upper bound: 0.1917883
time: 3.40 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1917875, upper bound: 0.1901748
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1917702, upper bound: 0.1901916
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1900105, upper bound: 0.1919525
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1899924, upper bound: 0.1919698
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1919689, upper bound: 0.1899933
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1919516, upper bound: 0.1900105
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1901920, upper bound: 0.1917710
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 6, lower bound: -0.1901739, upper bound: 0.1917883

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5455972, 0.5451188
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6699533, 0.6706362
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5210862, 0.5229151
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4948728, 0.4939753
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4618959, 0.4573326
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5064591, 0.5059755
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3599815, 0.3568721
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6178329, 0.6255844
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4335862, 0.4363277
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7191725, 0.7258503

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1846016, upper bound: 0.1881341
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1897433, upper bound: 0.1830010
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5472847, 0.5434313
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6705775, 0.6700120
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5210757, 0.5229259
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4948518, 0.4939963
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4595013, 0.4597273
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5063640, 0.5060709
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3585582, 0.3582953
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6190159, 0.6244013
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4315826, 0.4383312
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7239308, 0.7210920

Time for backsubstitution: 21.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1845970, upper bound: 0.1881474
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1897306, upper bound: 0.1830061
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5429345, 0.5477815
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6716089, 0.6689801
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5200710, 0.5239305
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4944656, 0.4943825
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4626675, 0.4565609
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5076836, 0.5047514
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3578967, 0.3589569
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6225250, 0.6208923
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4356605, 0.4342535
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7207942, 0.7242284

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1828236, upper bound: 0.1899118
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1879649, upper bound: 0.1847794
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5446221, 0.5460939
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6722336, 0.6683559
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5200601, 0.5239413
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4944441, 0.4944035
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4602728, 0.4589555
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5075883, 0.5048466
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3564733, 0.3603802
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6237085, 0.6197093
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4336568, 0.4362570
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7255530, 0.7194700

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1828186, upper bound: 0.1899258
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1879522, upper bound: 0.1847829
time: 3.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5457729, 0.5446222
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6683559, 0.6712008
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5220945, 0.5200601
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4944036, 0.4941412
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4589555, 0.4583708
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5048465, 0.5065455
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3601224, 0.3564733
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6184947, 0.6237085
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4345295, 0.4336568
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7192774, 0.7255530

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1847831, upper bound: 0.1879528
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1899247, upper bound: 0.1828194
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5474604, 0.5429347
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6689806, 0.6705766
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5220838, 0.5200708
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4943826, 0.4941622
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4565609, 0.4607654
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5047513, 0.5066407
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3586991, 0.3578967
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6196780, 0.6225250
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4325260, 0.4356605
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7240357, 0.7207947

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1847785, upper bound: 0.1879646
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1899120, upper bound: 0.1828245
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5431103, 0.5472848
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6700115, 0.6695452
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5210791, 0.5210755
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4939964, 0.4945484
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4597273, 0.4575992
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5060710, 0.5053213
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3580376, 0.3585582
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6231868, 0.6190159
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4366038, 0.4315826
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7208991, 0.7239311

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1830052, upper bound: 0.1897305
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1881462, upper bound: 0.1845979
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5447978, 0.5455973
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6706362, 0.6689210
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5210683, 0.5210862
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4939754, 0.4945694
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4573326, 0.4599937
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5059756, 0.5054166
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3566142, 0.3599815
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6243701, 0.6178329
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4346002, 0.4335862
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7256579, 0.7191727

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1830001, upper bound: 0.1897431
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1881336, upper bound: 0.1846026
time: 3.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1846016, upper bound: 0.1881341
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1897433, upper bound: 0.1830010
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1845970, upper bound: 0.1881474
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1897306, upper bound: 0.1830061
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1828236, upper bound: 0.1899118
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1879649, upper bound: 0.1847794
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1828186, upper bound: 0.1899258
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1879522, upper bound: 0.1847829
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1847831, upper bound: 0.1879528
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1899247, upper bound: 0.1828194
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1847785, upper bound: 0.1879646
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1899120, upper bound: 0.1828245
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1830052, upper bound: 0.1897305
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1881462, upper bound: 0.1845979
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1830001, upper bound: 0.1897431
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.04
Output dim: 6, lower bound: -0.1881336, upper bound: 0.1846026

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.76 + 412.07 = 467.83 seconds
