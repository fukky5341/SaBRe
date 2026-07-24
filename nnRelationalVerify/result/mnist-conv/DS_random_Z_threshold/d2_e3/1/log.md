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
execution time: IAR + RelationalAnalysis = 23.90 + 33.21 = 57.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1981739, upper bound: 0.1981734

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1963946
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963937, upper bound: 0.1981691
time: 2.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.04 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.04
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1963946
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.04
Output dim: 6, lower bound: -0.1963937, upper bound: 0.1981691

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5723343, 0.5696715
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6886687, 0.6903238
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5226028, 0.5215873
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4947634, 0.4943562
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4902284, 0.4910001
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5129950, 0.5142194
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3827226, 0.3806378
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5951710, 0.5998631
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4636562, 0.4657304
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7935801, 0.7952020

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1979873, upper bound: 0.1963923
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981657, upper bound: 0.1962122
time: 3.21 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5696714, 0.5723342
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6903243, 0.6886683
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5215871, 0.5226027
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4943562, 0.4947634
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4909999, 0.4902283
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5142195, 0.5129951
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3806379, 0.3827226
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5998633, 0.5951710
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4657304, 0.4636562
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7952018, 0.7935801

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1981666
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1979883
time: 3.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 6, lower bound: -0.1979873, upper bound: 0.1963923
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 6, lower bound: -0.1981657, upper bound: 0.1962122
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.67
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1981666
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.67
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

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 618

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1914

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1979725, upper bound: 0.1947344
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963288, upper bound: 0.1963774
time: 2.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 899

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1889928, upper bound: 0.1911369
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1930928, upper bound: 0.1870389
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1256

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868512, upper bound: 0.1953437
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933312, upper bound: 0.1886666
time: 3.13 seconds

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

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1940936, upper bound: 0.1956630
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1940869, upper bound: 0.1956670
time: 2.98 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1979725, upper bound: 0.1947344
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1963288, upper bound: 0.1963774
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1889928, upper bound: 0.1911369
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1930928, upper bound: 0.1870389
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1868512, upper bound: 0.1953437
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1933312, upper bound: 0.1886666
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1940936, upper bound: 0.1956630
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.35
Output dim: 6, lower bound: -0.1940869, upper bound: 0.1956670

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5722001, 0.5700231
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6895576, 0.6895952
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5216241, 0.5234710
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4949589, 0.4940976
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4921355, 0.4899682
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5140305, 0.5136429
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3826427, 0.3809477
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5945539, 0.6011388
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4626956, 0.4674412
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7934551, 0.7953715

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2826

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1977624, upper bound: 0.1945026
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1977609, upper bound: 0.1945222
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5721891, 0.5700339
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6895366, 0.6896162
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5216317, 0.5234634
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4949737, 0.4940828
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4921370, 0.4899670
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5140314, 0.5136420
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3826337, 0.3809568
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5945702, 0.6011226
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4626961, 0.4674406
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7934523, 0.7953744

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 549

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1926785, upper bound: 0.1893454
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1905116, upper bound: 0.1920302
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5736578, 0.5693977
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6880779, 0.6905169
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5224688, 0.5202204
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4952445, 0.4942276
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4891839, 0.4920660
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5128081, 0.5141553
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3824606, 0.3823632
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5951276, 0.5980947
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4637492, 0.4647231
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7943468, 0.7950203

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1143

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 618

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1885472, upper bound: 0.1910202
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1888744, upper bound: 0.1906048
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5722363, 0.5694957
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6881037, 0.6902976
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5226028, 0.5204448
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4944692, 0.4943562
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4891899, 0.4909937
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5123613, 0.5142194
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3827226, 0.3802348
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5940647, 0.5992010
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4635923, 0.4647871
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7935038, 0.7950966

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 899

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1903975, upper bound: 0.1858330
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1919074, upper bound: 0.1843213
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5471139, 0.5488846
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6898808, 0.6855273
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5173669, 0.5215715
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4910672, 0.4894471
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4919360, 0.4873137
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5129271, 0.5095780
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3574550, 0.3647014
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5898442, 0.5860023
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4644947, 0.4651113
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7823634, 0.7790444

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 206

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1806792, upper bound: 0.1892576
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1806473, upper bound: 0.1893189
time: 3.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5452318, 0.5502732
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6887798, 0.6865396
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5177011, 0.5211617
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4894254, 0.4910053
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4910254, 0.4881613
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5123737, 0.5100901
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3618493, 0.3599386
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5888181, 0.5867088
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4645042, 0.4650912
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7803688, 0.7808030

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1906575, upper bound: 0.1874353
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1921410, upper bound: 0.1859248
time: 3.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5580889, 0.5610237
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6833172, 0.6828923
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5214765, 0.5204829
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4941773, 0.4947519
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4830511, 0.4888592
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5071452, 0.5072957
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3747731, 0.3762100
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5872898, 0.5810544
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4639275, 0.4598560
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7930136, 0.7927001

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 549

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1844

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1256

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1848583, upper bound: 0.1926981
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1913120, upper bound: 0.1862861
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5585369, 0.5604932
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6838961, 0.6822257
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5204582, 0.5214835
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4941788, 0.4947500
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4885027, 0.4833177
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5079501, 0.5064057
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3742661, 0.3766466
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5862451, 0.5819354
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4628737, 0.4608814
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7943978, 0.7912869

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 206

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 618

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933163, upper bound: 0.1955165
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1939324, upper bound: 0.1953512
time: 3.01 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1977624, upper bound: 0.1945026
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1977609, upper bound: 0.1945222
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1926785, upper bound: 0.1893454
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1905116, upper bound: 0.1920302
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1885472, upper bound: 0.1910202
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1888744, upper bound: 0.1906048
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1903975, upper bound: 0.1858330
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1919074, upper bound: 0.1843213
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1806792, upper bound: 0.1892576
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1806473, upper bound: 0.1893189
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1906575, upper bound: 0.1874353
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1921410, upper bound: 0.1859248
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1848583, upper bound: 0.1926981
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1913120, upper bound: 0.1862861
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1933163, upper bound: 0.1955165
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.01
Output dim: 6, lower bound: -0.1939324, upper bound: 0.1953512

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5027150, 0.4967041
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6528752, 0.6538672
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.4871495, 0.4894837
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4160995, 0.4102457
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4880552, 0.4842724
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.4991988, 0.4978923
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3751653, 0.3727312
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5904400, 0.5969372
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4626927, 0.4674540
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7513871, 0.7446880

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1907704, upper bound: 0.1924462
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1956866, upper bound: 0.1875449
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.4990586, 0.5005490
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6539638, 0.6529336
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.4878421, 0.4889891
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4111221, 0.4154826
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4864802, 0.4858867
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.4982804, 0.4988896
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3744481, 0.3734792
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5904016, 0.5970085
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4627090, 0.4674381
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7427688, 0.7535229

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 549
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1914

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2826

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1975365, upper bound: 0.1934040
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1966405, upper bound: 0.1942963
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5720663, 0.5696514
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6896448, 0.6896601
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5209999, 0.5232956
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4947519, 0.4951478
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4921660, 0.4897809
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5133882, 0.5133971
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3822298, 0.3800521
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5936499, 0.6008592
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4626336, 0.4673933
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7923574, 0.7947748

Time for backsubstitution: 22.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2865

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1924262, upper bound: 0.1867897
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1898157, upper bound: 0.1890792
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5721583, 0.5699003
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6897011, 0.6897030
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5214562, 0.5234340
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4950666, 0.4938757
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4919496, 0.4899616
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5137856, 0.5136496
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3825815, 0.3805437
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5942907, 0.6010773
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4626482, 0.4674579
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7928557, 0.7953942

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2515
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1405
type: DSZ, layer: 3, pos: 1255
type: DSZ, layer: 3, pos: 206
type: DSZ, layer: 3, pos: 2865
type: DSZ, layer: 3, pos: 2613
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1914
type: DSZ, layer: 3, pos: 1193
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 618
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2826
type: DSZ, layer: 3, pos: 1256
type: DSZ, layer: 3, pos: 2124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2515

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1825397, upper bound: 0.1900814
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1885632, upper bound: 0.1842435
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5500271, 0.5469579
1: -12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6883764, 0.6905823
2: -7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5153308, 0.5134358
3: -11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4963014, 0.4956478
4: -2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4892063, 0.4910148
5: -5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5107890, 0.5116874
6: 7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3826432, 0.3804209
7: -17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.5932488, 0.5973740
8: -3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4475940, 0.4526783
9: -10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7894602, 0.7911062

Time for backsubstitution: 22.75 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.10 + 558.98 = 616.08 seconds
