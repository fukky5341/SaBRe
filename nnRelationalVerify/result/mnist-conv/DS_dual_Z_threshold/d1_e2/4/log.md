## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.214332525


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3835130, 0.3835125)
1: (-4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3126643, 0.3126643)
2: (7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3854012, 0.3854012)
3: (-2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3425937, 0.3425937)
4: (-12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3527765, 0.3527765)
5: (-10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3518991, 0.3518991)
6: (-8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2934105, 0.2934110)
7: (-8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3171108, 0.3171108)
8: (-2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2629416, 0.2629416)
9: (-12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3359382, 0.3359382)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.86 + 34.58 = 56.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.2164975, upper bound: 0.2164976

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 484

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164963, upper bound: 0.2155334
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155332, upper bound: 0.2164964
time: 3.86 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.55 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.55
Output dim: 2, lower bound: -0.2164963, upper bound: 0.2155334
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.55
Output dim: 2, lower bound: -0.2155332, upper bound: 0.2164964

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3859200, 0.3867102
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3079011, 0.3094153
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825946, 0.3816354
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3380218, 0.3365016
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495221, 0.3503356
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449554, 0.3466873
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2895050, 0.2904801
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153603, 0.3160098
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583811, 0.2567272
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3336241, 0.3344412

Time for backsubstitution: 20.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164961, upper bound: 0.2147493
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2157123, upper bound: 0.2155333
time: 4.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3867102, 0.3859200
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3094153, 0.3079011
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3816352, 0.3825943
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3365016, 0.3380218
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3503356, 0.3495221
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3466873, 0.3449554
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2904801, 0.2895050
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3160098, 0.3153601
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567272, 0.2583811
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3344412, 0.3336241

Time for backsubstitution: 19.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155331, upper bound: 0.2157124
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147491, upper bound: 0.2164960
time: 3.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.62 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 2, lower bound: -0.2164961, upper bound: 0.2147493
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 2, lower bound: -0.2157123, upper bound: 0.2155333
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 2, lower bound: -0.2155331, upper bound: 0.2157124
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.62
Output dim: 2, lower bound: -0.2147491, upper bound: 0.2164960

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3859196, 0.3867097
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3078995, 0.3094130
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825912, 0.3816321
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3380198, 0.3364992
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495214, 0.3503351
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449593, 0.3466897
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2895026, 0.2904785
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153601, 0.3160098
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583807, 0.2567272
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3336234, 0.3344407

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2162282, upper bound: 0.2146865
time: 5.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2146891
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3859196, 0.3867097
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3078988, 0.3094134
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825903, 0.3816326
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3380194, 0.3364997
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495216, 0.3503351
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449583, 0.3466907
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2895036, 0.2904780
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153601, 0.3160098
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583811, 0.2567267
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3336236, 0.3344405

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154441, upper bound: 0.2154707
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146979, upper bound: 0.2154731
time: 3.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3867097, 0.3859196
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3094134, 0.3078988
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3816328, 0.3825905
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3364997, 0.3380194
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3503351, 0.3495216
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3466907, 0.3449583
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2904782, 0.2895036
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3160095, 0.3153603
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567267, 0.2583811
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3344405, 0.3336236

Time for backsubstitution: 21.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154729, upper bound: 0.2146978
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154705, upper bound: 0.2154440
time: 4.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3867097, 0.3859196
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3094130, 0.3078995
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3816319, 0.3825915
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3364992, 0.3380198
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3503354, 0.3495214
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3466897, 0.3449593
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2904782, 0.2895029
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3160095, 0.3153601
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567272, 0.2583807
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3344407, 0.3336234

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146890, upper bound: 0.2154817
time: 5.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146866, upper bound: 0.2162280
time: 6.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.95 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2162282, upper bound: 0.2146865
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2146891
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2154441, upper bound: 0.2154707
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2146979, upper bound: 0.2154731
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2154729, upper bound: 0.2146978
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2154705, upper bound: 0.2154440
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2146890, upper bound: 0.2154817
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.95
Output dim: 2, lower bound: -0.2146866, upper bound: 0.2162280

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3859525, 0.3866158
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3079226, 0.3093436
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3827457, 0.3816292
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3380380, 0.3364449
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3496363, 0.3503325
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3445692, 0.3468280
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2892051, 0.2905798
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3155932, 0.3160052
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583802, 0.2567592
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3336782, 0.3342857

Time for backsubstitution: 20.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2145658, upper bound: 0.2120553
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2137039, upper bound: 0.2130866
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3858261, 0.3867097
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3078301, 0.3094130
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825884, 0.3816321
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3379655, 0.3364992
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495188, 0.3503351
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449593, 0.3462996
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2895026, 0.2901804
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153558, 0.3160098
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583807, 0.2567265
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3334684, 0.3344407

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2138809, upper bound: 0.2120581
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2128492, upper bound: 0.2130879
time: 4.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3859525, 0.3866158
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3079219, 0.3093443
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3827448, 0.3816297
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3380375, 0.3364453
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3496366, 0.3503323
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3445683, 0.3468285
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2892056, 0.2905791
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3155932, 0.3160050
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583804, 0.2567587
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3336785, 0.3342855

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2137819, upper bound: 0.2128395
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2129198, upper bound: 0.2138704
time: 4.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3858261, 0.3867097
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3078296, 0.3094134
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825884, 0.3816326
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3379650, 0.3364997
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495188, 0.3503351
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449583, 0.3463006
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2895036, 0.2901800
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153558, 0.3160098
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583811, 0.2567263
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3334687, 0.3344405

Time for backsubstitution: 23.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2130971, upper bound: 0.2128422
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2120650, upper bound: 0.2138717
time: 5.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3864942, 0.3858261
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3092537, 0.3078296
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3817863, 0.3825881
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3365173, 0.3379650
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3502178, 0.3495188
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3463006, 0.3450947
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2901797, 0.2896042
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3157792, 0.3153555
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567263, 0.2583489
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3340845, 0.3334687

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2138718, upper bound: 0.2120650
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2128421, upper bound: 0.2130973
time: 3.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3866158, 0.3859196
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3093443, 0.3078988
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3816299, 0.3825905
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3364453, 0.3380194
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3503323, 0.3495216
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3466907, 0.3445683
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2904782, 0.2892056
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3160052, 0.3153603
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567267, 0.2583804
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3342855, 0.3336236

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2138702, upper bound: 0.2129197
time: 5.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2128393, upper bound: 0.2137821
time: 3.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3864942, 0.3858261
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3092530, 0.3078301
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3817854, 0.3825886
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3365169, 0.3379655
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3502181, 0.3495188
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3462996, 0.3450956
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2901807, 0.2896037
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3157792, 0.3153555
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567265, 0.2583485
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3340847, 0.3334684

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2130880, upper bound: 0.2128491
time: 7.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2120579, upper bound: 0.2138811
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3866158, 0.3859196
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3093436, 0.3078995
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3816290, 0.3825915
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3364449, 0.3380198
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3503325, 0.3495214
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3466897, 0.3445692
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2904782, 0.2892048
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3160052, 0.3153601
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2567272, 0.2583802
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3342857, 0.3336234

Time for backsubstitution: 22.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 654

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2130864, upper bound: 0.2137039
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2120551, upper bound: 0.2145660
time: 3.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2145658, upper bound: 0.2120553
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2137039, upper bound: 0.2130866
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2138809, upper bound: 0.2120581
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2128492, upper bound: 0.2130879
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2137819, upper bound: 0.2128395
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2129198, upper bound: 0.2138704
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2130971, upper bound: 0.2128422
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2120650, upper bound: 0.2138717
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2138718, upper bound: 0.2120650
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2128421, upper bound: 0.2130973
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2138702, upper bound: 0.2129197
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2128393, upper bound: 0.2137821
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2130880, upper bound: 0.2128491
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2120579, upper bound: 0.2138811
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2130864, upper bound: 0.2137039
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.91
Output dim: 2, lower bound: -0.2120551, upper bound: 0.2145660

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3880467, 0.3853297
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3076103, 0.3091791
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3811064, 0.3784249
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3374190, 0.3355150
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3442185, 0.3462648
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3432612, 0.3450961
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2890763, 0.2903886
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3074870, 0.3077888
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2558990, 0.2552080
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3325918, 0.3336761

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2138040, upper bound: 0.2070438
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2095173, upper bound: 0.2113058
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3853297, 0.3880153
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3091791, 0.3075874
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3784246, 0.3810642
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3355150, 0.3374009
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3462648, 0.3441036
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449583, 0.3432612
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2902875, 0.2890766
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3077888, 0.3074269
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2551761, 0.2558990
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3336761, 0.3325365

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2113056, upper bound: 0.2095175
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2070436, upper bound: 0.2138043
time: 4.18 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.92 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 2, lower bound: -0.2138040, upper bound: 0.2070438
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 2, lower bound: -0.2095173, upper bound: 0.2113058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 2, lower bound: -0.2113056, upper bound: 0.2095175
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.92
Output dim: 2, lower bound: -0.2070436, upper bound: 0.2138043

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.44 + 494.04 = 550.48 seconds
