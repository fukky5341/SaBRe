## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.061818120000000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2605984, 0.2605989)
1: (-3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1932678, 0.1932678)
2: (-8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2409225, 0.2409225)
3: (-1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2066152, 0.2066151)
4: (-4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1671135, 0.1671135)
5: (-6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1912547, 0.1912547)
6: (-15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2285557, 0.2285556)
7: (4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1057246, 0.1057247)
8: (-4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2120063, 0.2120063)
9: (-0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1830323, 0.1830324)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.54 + 32.88 = 55.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0686868, upper bound: 0.0686868

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1852

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671821, upper bound: 0.0675171
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675165, upper bound: 0.0671816
time: 2.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.13 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.13
Output dim: 7, lower bound: -0.0671821, upper bound: 0.0675171
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.13
Output dim: 7, lower bound: -0.0675165, upper bound: 0.0671816

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2354115, 0.2330308
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1893427, 0.1897225
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2299738, 0.2296407
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2067833, 0.2055140
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1624963, 0.1639831
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1809344, 0.1817029
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2124245, 0.2123877
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0999052, 0.1005909
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1984391, 0.1973155
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1828767, 0.1834140

Time for backsubstitution: 8.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 312

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 558

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0668757, upper bound: 0.0673555
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0670207, upper bound: 0.0672105
time: 2.68 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2330309, 0.2354114
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1897225, 0.1893427
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2296405, 0.2299738
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2055140, 0.2067833
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1639831, 0.1624963
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1817029, 0.1809344
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2123878, 0.2124246
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1005909, 0.0999052
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1973155, 0.1984391
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1834141, 0.1828766

Time for backsubstitution: 8.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 962

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0671739, upper bound: 0.0665009
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0668359, upper bound: 0.0668389
time: 2.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.37 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.37
Output dim: 7, lower bound: -0.0668757, upper bound: 0.0673555
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.37
Output dim: 7, lower bound: -0.0670207, upper bound: 0.0672105
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.37
Output dim: 7, lower bound: -0.0671739, upper bound: 0.0665009
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.37
Output dim: 7, lower bound: -0.0668359, upper bound: 0.0668389

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2322648, 0.2301874
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1890187, 0.1891849
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2294104, 0.2293117
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2046404, 0.2027552
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1618212, 0.1636523
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1805358, 0.1813426
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2121973, 0.2118483
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0997195, 0.1004384
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1974748, 0.1965141
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1800629, 0.1799674

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1955

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1852

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654822, upper bound: 0.0670852
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666049, upper bound: 0.0659616
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2325678, 0.2298841
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1888051, 0.1893988
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2296450, 0.2290773
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2040243, 0.2033710
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1621655, 0.1633080
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1805742, 0.1813043
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2118852, 0.2121606
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0997527, 0.1004052
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1976378, 0.1963512
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1794299, 0.1806002

Time for backsubstitution: 8.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0664343, upper bound: 0.0649653
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650859, upper bound: 0.0666642
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2324681, 0.2344711
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1885376, 0.1878052
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2290068, 0.2292886
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2051084, 0.2063677
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1639568, 0.1624877
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1810006, 0.1802888
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2104237, 0.2110193
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1001273, 0.0993787
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1968340, 0.1981902
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1833206, 0.1828139

Time for backsubstitution: 7.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1955

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666284, upper bound: 0.0645941
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0649329, upper bound: 0.0659145
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2320907, 0.2348487
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1881852, 0.1881576
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2289553, 0.2293279
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2050984, 0.2063776
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1639745, 0.1624700
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1810573, 0.1802321
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2109826, 0.2104607
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1000645, 0.0994415
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1970667, 0.1979576
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1833514, 0.1827836

Time for backsubstitution: 8.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 557

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0666428, upper bound: 0.0628239
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0629146, upper bound: 0.0666456
time: 2.71 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.96 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0654822, upper bound: 0.0670852
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0666049, upper bound: 0.0659616
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0664343, upper bound: 0.0649653
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0650859, upper bound: 0.0666642
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0666284, upper bound: 0.0645941
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0649329, upper bound: 0.0659145
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0666428, upper bound: 0.0628239
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.96
Output dim: 7, lower bound: -0.0629146, upper bound: 0.0666456

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2333182, 0.2311730
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1895354, 0.1891769
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2258735, 0.2260134
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2049844, 0.2030383
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1608849, 0.1631234
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1803875, 0.1810539
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2122970, 0.2119424
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0975544, 0.0987200
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1984224, 0.1975302
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1783161, 0.1787032

Time for backsubstitution: 8.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 312

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637248, upper bound: 0.0652910
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637248, upper bound: 0.0652910
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2332505, 0.2312407
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1890109, 0.1897012
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2261121, 0.2257750
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2049234, 0.2030993
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1612923, 0.1627159
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1802471, 0.1811943
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2122915, 0.2119477
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0980011, 0.0982733
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1984910, 0.1974617
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1787987, 0.1782206

Time for backsubstitution: 8.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1485

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658532, upper bound: 0.0652617
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658590, upper bound: 0.0652545
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2321203, 0.2295947
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1884546, 0.1888903
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2288535, 0.2283862
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2028763, 0.2023168
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1601988, 0.1616724
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1755006, 0.1768917
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2108052, 0.2112889
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0994161, 0.0996617
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1970626, 0.1956753
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1708189, 0.1758003

Time for backsubstitution: 8.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1852

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1485

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662414, upper bound: 0.0648994
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663690, upper bound: 0.0647726
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2322783, 0.2294750
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1884491, 0.1890482
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2289538, 0.2282858
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2030373, 0.2022231
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1605299, 0.1615030
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1761615, 0.1762309
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2111993, 0.2110804
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0990092, 0.1001696
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1969618, 0.1958692
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1746436, 0.1719894

Time for backsubstitution: 8.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 312

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0649056, upper bound: 0.0666627
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650844, upper bound: 0.0664842
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2320589, 0.2341819
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1881869, 0.1874491
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2282157, 0.2285975
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2039602, 0.2053807
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1621517, 0.1608521
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1759273, 0.1758762
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2093437, 0.2103335
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0998916, 0.0986351
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1963522, 0.1975145
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1747092, 0.1780269

Time for backsubstitution: 7.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1852

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652317, upper bound: 0.0643238
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663590, upper bound: 0.0631339
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2321788, 0.2340236
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1880291, 0.1874546
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2283158, 0.2284971
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2040541, 0.2052196
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1623211, 0.1605210
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1765881, 0.1752155
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2095520, 0.2099392
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993837, 0.0990421
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1961582, 0.1976154
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1785201, 0.1742023

Time for backsubstitution: 7.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647527, upper bound: 0.0659125
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0649312, upper bound: 0.0657344
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2323091, 0.2350912
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1842015, 0.1846772
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2288280, 0.2290084
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1958860, 0.1972362
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1414014, 0.1401098
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1807304, 0.1799837
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1930475, 0.1946770
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993296, 0.0979725
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1959076, 0.1968064
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1530820, 0.1549323

Time for backsubstitution: 7.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 311

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0659166, upper bound: 0.0620773
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0659203, upper bound: 0.0620716
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2323329, 0.2350671
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1847045, 0.1841685
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2286360, 0.2292006
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1960429, 0.1971651
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1416144, 0.1397352
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1808090, 0.1799052
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1951988, 0.1926711
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0985954, 0.0987675
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1959152, 0.1967986
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1555000, 0.1529908

Time for backsubstitution: 8.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1955

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 429

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0590153, upper bound: 0.0627466
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0590153, upper bound: 0.0627466
time: 2.70 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0637248, upper bound: 0.0652910
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0637248, upper bound: 0.0652910
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0658532, upper bound: 0.0652617
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0658590, upper bound: 0.0652545
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0662414, upper bound: 0.0648994
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0663690, upper bound: 0.0647726
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0649056, upper bound: 0.0666627
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0650844, upper bound: 0.0664842
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0652317, upper bound: 0.0643238
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0663590, upper bound: 0.0631339
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0647527, upper bound: 0.0659125
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0649312, upper bound: 0.0657344
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0659166, upper bound: 0.0620773
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0659203, upper bound: 0.0620716
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0590153, upper bound: 0.0627466
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0590153, upper bound: 0.0627466

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2326994, 0.2306762
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1893516, 0.1885287
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2251742, 0.2253358
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2047517, 0.2026334
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1603708, 0.1625130
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1805459, 0.1804994
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2116992, 0.2117926
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0964820, 0.0980649
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1967894, 0.1958370
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1783811, 0.1781410

Time for backsubstitution: 8.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0635448, upper bound: 0.0652895
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637233, upper bound: 0.0651109
time: 2.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2333182, 0.2305543
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1888871, 0.1891769
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2258735, 0.2253139
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2049844, 0.2028056
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1608849, 0.1626093
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1798331, 0.1810539
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2122970, 0.2113448
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0968994, 0.0987200
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1984224, 0.1958971
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1777538, 0.1787032

Time for backsubstitution: 7.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 326

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630176, upper bound: 0.0645368
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630322, upper bound: 0.0645341
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2229681, 0.2189870
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1818146, 0.1797875
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2245493, 0.2241126
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2042259, 0.2027339
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1540734, 0.1540409
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1695950, 0.1704541
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2075160, 0.2065977
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0927681, 0.0929238
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1863804, 0.1871290
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1735640, 0.1723925

Time for backsubstitution: 8.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 557

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650328, upper bound: 0.0647510
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0653739, upper bound: 0.0643064
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2209971, 0.2209581
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1789137, 0.1825048
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2244499, 0.2241665
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2045823, 0.2024019
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1526173, 0.1553729
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1695070, 0.1705421
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2069508, 0.2071720
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0926359, 0.0930403
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1881582, 0.1855969
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1729705, 0.1729854

Time for backsubstitution: 7.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650378, upper bound: 0.0647438
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0653795, upper bound: 0.0642993
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2320894, 0.2294164
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1883669, 0.1888125
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2286336, 0.2279609
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2028208, 0.2022533
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1595345, 0.1611899
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1734101, 0.1751602
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2096627, 0.2092191
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993407, 0.0996149
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1969178, 0.1954274
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1707581, 0.1757169

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1852

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 962

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658984, upper bound: 0.0642186
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0655604, upper bound: 0.0645565
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2319416, 0.2295640
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1883767, 0.1888027
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2284284, 0.2281662
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2028127, 0.2022614
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1597164, 0.1610080
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1737692, 0.1748011
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2087352, 0.2101463
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993693, 0.0995864
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1968148, 0.1955303
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1707355, 0.1757394

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 962

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660254, upper bound: 0.0640909
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656878, upper bound: 0.0644294
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2322768, 0.2294719
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1883647, 0.1889789
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2289457, 0.2282741
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2030057, 0.2021927
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1605024, 0.1614740
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1761364, 0.1762086
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2111754, 0.2110530
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0990070, 0.1001679
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1969510, 0.1958598
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1745716, 0.1719097

Time for backsubstitution: 7.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 429

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0611869, upper bound: 0.0626811
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0611869, upper bound: 0.0626811
time: 2.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2322756, 0.2294731
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1883798, 0.1889639
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2289422, 0.2282779
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2030071, 0.2021912
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1605010, 0.1614755
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1761391, 0.1762057
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2111719, 0.2110565
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0990074, 0.1001675
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1969525, 0.1958585
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1745638, 0.1719174

Time for backsubstitution: 7.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0639619, upper bound: 0.0650066
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0636019, upper bound: 0.0653656
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2331120, 0.2351675
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1887074, 0.1874455
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2247119, 0.2253325
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2042828, 0.2056423
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1612159, 0.1603238
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1757894, 0.1755980
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2094185, 0.2104030
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0976831, 0.0968732
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1972996, 0.1985303
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1729585, 0.1767590

Time for backsubstitution: 8.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650516, upper bound: 0.0643223
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652301, upper bound: 0.0641438
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2330445, 0.2352350
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1881831, 0.1879698
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2247815, 0.2250938
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2042220, 0.2056789
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1616234, 0.1599163
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1756489, 0.1757384
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2094133, 0.2104071
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0981297, 0.0964312
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1973680, 0.1984619
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1734413, 0.1762764

Time for backsubstitution: 8.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2568

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662053, upper bound: 0.0629617
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662136, upper bound: 0.0629526
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2321769, 0.2340207
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1879444, 0.1873851
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2283070, 0.2284847
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2040223, 0.2051895
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1622938, 0.1604923
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1765629, 0.1751931
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2095282, 0.2099118
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993816, 0.0990404
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1961476, 0.1976061
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1784482, 0.1741228

Time for backsubstitution: 8.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 312

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1485

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0645602, upper bound: 0.0658478
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0646871, upper bound: 0.0657198
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2321759, 0.2340219
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1879597, 0.1873703
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2283034, 0.2284883
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2040237, 0.2051880
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1622924, 0.1604937
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1765658, 0.1751903
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2095246, 0.2099153
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0993820, 0.0990400
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1961490, 0.1976047
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1784406, 0.1741306

Time for backsubstitution: 8.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 429

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0609476, upper bound: 0.0620136
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0609476, upper bound: 0.0620136
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2220736, 0.2228843
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1765420, 0.1741168
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2269642, 0.2270910
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1951960, 0.1969025
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1350138, 0.1323844
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1707330, 0.1698983
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1879456, 0.1890101
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0947731, 0.0932838
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1838430, 0.1862731
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1482399, 0.1494972

Time for backsubstitution: 7.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 311

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2568

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658014, upper bound: 0.0618919
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658109, upper bound: 0.0618800
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2201024, 0.2248555
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1737454, 0.1770177
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2269106, 0.2271625
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1955523, 0.1965462
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1336759, 0.1338404
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1706449, 0.1699863
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1873739, 0.1895751
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0946455, 0.0934160
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1853741, 0.1847968
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1476469, 0.1500906

Time for backsubstitution: 9.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0657403, upper bound: 0.0620701
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0659188, upper bound: 0.0618916
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2322512, 0.2350934
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1846976, 0.1841651
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2285089, 0.2291424
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1960096, 0.1973833
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1415044, 0.1384271
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1807436, 0.1805331
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1951729, 0.1927533
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0987741, 0.0986822
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1959001, 0.1967306
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1553123, 0.1528964

Time for backsubstitution: 9.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 558

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0589028, upper bound: 0.0625851
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0587009, upper bound: 0.0624208
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2323329, 0.2349851
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1847045, 0.1841615
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2286360, 0.2290735
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1960429, 0.1971319
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1416144, 0.1396252
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1808090, 0.1798399
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1951988, 0.1926452
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0985101, 0.0987675
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1959152, 0.1967834
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1554056, 0.1529908

Time for backsubstitution: 9.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 558
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0590142, upper bound: 0.0626609
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0589296, upper bound: 0.0627455
time: 2.85 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0635448, upper bound: 0.0652895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0637233, upper bound: 0.0651109
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0630176, upper bound: 0.0645368
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0630322, upper bound: 0.0645341
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0650328, upper bound: 0.0647510
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0653739, upper bound: 0.0643064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0650378, upper bound: 0.0647438
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0653795, upper bound: 0.0642993
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0658984, upper bound: 0.0642186
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0655604, upper bound: 0.0645565
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0660254, upper bound: 0.0640909
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0656878, upper bound: 0.0644294
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0611869, upper bound: 0.0626811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0611869, upper bound: 0.0626811
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0639619, upper bound: 0.0650066
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0636019, upper bound: 0.0653656
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0650516, upper bound: 0.0643223
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0652301, upper bound: 0.0641438
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0662053, upper bound: 0.0629617
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0662136, upper bound: 0.0629526
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0645602, upper bound: 0.0658478
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0646871, upper bound: 0.0657198
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0609476, upper bound: 0.0620136
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0609476, upper bound: 0.0620136
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0658014, upper bound: 0.0618919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0658109, upper bound: 0.0618800
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0657403, upper bound: 0.0620701
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0659188, upper bound: 0.0618916
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0589028, upper bound: 0.0625851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0587009, upper bound: 0.0624208
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0590142, upper bound: 0.0626609
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.70
Output dim: 7, lower bound: -0.0589296, upper bound: 0.0627455

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2326975, 0.2306731
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1892674, 0.1884594
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2251658, 0.2253239
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2047199, 0.2026031
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1603433, 0.1624841
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1805209, 0.1804772
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2116747, 0.2117645
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0964798, 0.0980631
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1967788, 0.1958278
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1783093, 0.1780616

Time for backsubstitution: 9.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 557

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 429

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0598262, upper bound: 0.0613075
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0598262, upper bound: 0.0613075
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2326963, 0.2306743
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1892824, 0.1884446
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2251623, 0.2253275
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2047213, 0.2026017
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1603419, 0.1624856
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1805236, 0.1804744
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2116714, 0.2117680
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0964802, 0.0980628
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1967802, 0.1958265
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1783015, 0.1780694

Time for backsubstitution: 9.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0629044, upper bound: 0.0646004
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0632450, upper bound: 0.0641564
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2230357, 0.2183002
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1816911, 0.1790797
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2242651, 0.2236519
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2042869, 0.2024646
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1536659, 0.1539344
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1691810, 0.1703137
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2075212, 0.2059954
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0916666, 0.0933547
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1866134, 0.1855644
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1725186, 0.1728752

Time for backsubstitution: 9.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0621981, upper bound: 0.0640257
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0625388, upper bound: 0.0635819
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2210646, 0.2202712
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1789738, 0.1819805
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2242115, 0.2237515
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2046188, 0.2021084
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1522099, 0.1552664
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1690930, 0.1704018
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2069551, 0.2065688
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0915502, 0.0934870
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1880897, 0.1837308
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1719252, 0.1734681

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 429

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0591348, upper bound: 0.0606362
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0591348, upper bound: 0.0606362
time: 3.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2182505, 0.2150548
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1790774, 0.1747743
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2212086, 0.2204800
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1677773, 0.1626419
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1485575, 0.1488010
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1555416, 0.1574330
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1961232, 0.1971176
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0899252, 0.0904065
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1720586, 0.1731009
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1498321, 0.1520588

Time for backsubstitution: 8.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 312

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 962

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0646904, upper bound: 0.0640836
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0643803, upper bound: 0.0643993
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2190359, 0.2142694
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1768010, 0.1768709
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2209167, 0.2207720
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1641347, 0.1662852
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1488335, 0.1485251
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1565742, 0.1564006
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1977675, 0.1952050
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0902508, 0.0900811
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1717279, 0.1728072
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1532302, 0.1485779

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 2568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647876, upper bound: 0.0621360
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0634504, upper bound: 0.0637610
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2162795, 0.2170258
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1761765, 0.1774913
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2211094, 0.2205338
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1681337, 0.1623098
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1471015, 0.1501330
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1554535, 0.1575210
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1955581, 0.1976919
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0897930, 0.0905231
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1738362, 0.1715689
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1492386, 0.1526518

Time for backsubstitution: 9.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2341

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1955

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0648574, upper bound: 0.0647419
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650360, upper bound: 0.0645633
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2170649, 0.2162406
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1739001, 0.1795882
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2208173, 0.2208259
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.1644912, 0.1659533
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1473774, 0.1498571
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1564861, 0.1564887
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.1972024, 0.1957793
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0901186, 0.0901976
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1735055, 0.1712751
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1526369, 0.1491710

Time for backsubstitution: 9.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1955
type: DSZ, layer: 3, pos: 2568
type: DSZ, layer: 3, pos: 312
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1485

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0642575, upper bound: 0.0631542
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0639016, upper bound: 0.0632280
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2315265, 0.2284760
1: -3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1871817, 0.1872752
2: -8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2279878, 0.2272757
3: -1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2024150, 0.2018377
4: -4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1595080, 0.1611813
5: -6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1727077, 0.1745146
6: -15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2076986, 0.2078139
7: 4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.0988771, 0.0990884
8: -4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.1964363, 0.1951785
9: -0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1706649, 0.1756542

Time for backsubstitution: 9.08 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.42 + 549.00 = 604.42 seconds
