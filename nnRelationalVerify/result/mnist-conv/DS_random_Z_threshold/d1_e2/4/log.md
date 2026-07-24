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
execution time: IAR + RelationalAnalysis = 24.04 + 33.47 = 57.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.2164975, upper bound: 0.2164976

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164974, upper bound: 0.2157136
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2157135, upper bound: 0.2164975
time: 3.39 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 2, lower bound: -0.2164974, upper bound: 0.2157136
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 2, lower bound: -0.2157135, upper bound: 0.2164975

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3835111, 0.3835115
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3126628, 0.3126621
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3853989, 0.3853984
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3425927, 0.3425927
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3527756, 0.3527756
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3519011, 0.3519001
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2934089, 0.2934093
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3171103, 0.3171105
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2629416, 0.2629421
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3359375, 0.3359377

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 484

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2164961, upper bound: 0.2147493
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2155331, upper bound: 0.2157124
time: 4.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3835115, 0.3835111
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3126621, 0.3126628
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3853984, 0.3853989
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3425927, 0.3425927
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3527756, 0.3527756
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3519001, 0.3519011
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2934093, 0.2934086
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3171103, 0.3171103
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2629421, 0.2629416
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3359377, 0.3359375

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 484

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2157123, upper bound: 0.2155333
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2147491, upper bound: 0.2164960
time: 3.93 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.66
Output dim: 2, lower bound: -0.2164961, upper bound: 0.2147493
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.66
Output dim: 2, lower bound: -0.2155331, upper bound: 0.2157124
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.66
Output dim: 2, lower bound: -0.2157123, upper bound: 0.2155333
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.66
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

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2162282, upper bound: 0.2146865
time: 5.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2146891
time: 4.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 23.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154729, upper bound: 0.2146978
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154705, upper bound: 0.2154440
time: 4.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154441, upper bound: 0.2154707
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146979, upper bound: 0.2154731
time: 4.08 seconds

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

Time for backsubstitution: 22.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 79

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146890, upper bound: 0.2154817
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146866, upper bound: 0.2162280
time: 6.25 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.63 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2162282, upper bound: 0.2146865
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2154818, upper bound: 0.2146891
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2154729, upper bound: 0.2146978
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2154705, upper bound: 0.2154440
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2154441, upper bound: 0.2154707
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2146979, upper bound: 0.2154731
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.63
Output dim: 2, lower bound: -0.2146890, upper bound: 0.2154817
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.63
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

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2474

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154761, upper bound: 0.2096674
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2112112, upper bound: 0.2139339
time: 4.55 seconds

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

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 963

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154407, upper bound: 0.2078939
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2086867, upper bound: 0.2146480
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2333

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 963

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2154317, upper bound: 0.2079030
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2086777, upper bound: 0.2146569
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2468

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2136741, upper bound: 0.2134066
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2134987, upper bound: 0.2136176
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3124

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2123102, upper bound: 0.2125458
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2124112, upper bound: 0.2123733
time: 4.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 963

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146567, upper bound: 0.2086779
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2079028, upper bound: 0.2154319
time: 3.91 seconds

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

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1432

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2333

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2048307, upper bound: 0.2129607
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2121680, upper bound: 0.2056265
time: 3.76 seconds

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

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2333

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1823

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2112833, upper bound: 0.2093068
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2077659, upper bound: 0.2128288
time: 6.96 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2154761, upper bound: 0.2096674
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2112112, upper bound: 0.2139339
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2154407, upper bound: 0.2078939
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2086867, upper bound: 0.2146480
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2154317, upper bound: 0.2079030
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2086777, upper bound: 0.2146569
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2136741, upper bound: 0.2134066
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2134987, upper bound: 0.2136176
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2123102, upper bound: 0.2125458
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2124112, upper bound: 0.2123733
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2146567, upper bound: 0.2086779
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2079028, upper bound: 0.2154319
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2048307, upper bound: 0.2129607
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2121680, upper bound: 0.2056265
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2112833, upper bound: 0.2093068
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.01
Output dim: 2, lower bound: -0.2077659, upper bound: 0.2128288

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3865218, 0.3873196
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.2994113, 0.3031552
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3814998, 0.3801451
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3324833, 0.3313031
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3371665, 0.3399181
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3422551, 0.3424544
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2875876, 0.2871647
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3149390, 0.3158734
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2569246, 0.2555561
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3292859, 0.3305795

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2333

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 409

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2093619, upper bound: 0.2080973
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2139559, upper bound: 0.2038594
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3858175, 0.3866801
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3078187, 0.3094780
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825846, 0.3815997
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3376465, 0.3363104
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495169, 0.3503478
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3448052, 0.3461518
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2894702, 0.2901330
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153033, 0.3159778
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2582247, 0.2566943
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3334634, 0.3344386

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2143
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 563

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2146876, upper bound: 0.2028725
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2104216, upper bound: 0.2071406
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3858261, 0.3867016
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3078301, 0.3094013
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3825884, 0.3816278
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3377762, 0.3364992
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3495188, 0.3503337
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3449593, 0.3461456
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2895026, 0.2901478
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3153234, 0.3160098
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2583485, 0.2567265
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3334661, 0.3344407

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1823
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2348
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2543
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1513
type: DSZ, layer: 3, pos: 2143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2468

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2068893, upper bound: 0.2126664
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2067158, upper bound: 0.2128466
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.5146112, -5.0062685, -5.5146112, -5.0062685, -0.3864856, 0.3857965
1: -4.2234378, -3.6703835, -4.2234378, -3.6703835, -0.3092422, 0.3078947
2: 7.2144527, 7.8699956, 7.2144527, 7.8699956, -0.3817816, 0.3825557
3: -2.3853123, -1.8777944, -2.3853123, -1.8777944, -0.3361983, 0.3377757
4: -12.7614355, -12.0576487, -12.7614355, -12.0576487, -0.3502162, 0.3495314
5: -10.7035141, -10.1200981, -10.7035141, -10.1200981, -0.3461466, 0.3449469
6: -8.0690289, -7.5780745, -8.0690289, -7.5780745, -0.2901473, 0.2895567
7: -8.1379309, -7.5549507, -8.1379309, -7.5549507, -0.3157268, 0.3153236
8: -2.1984138, -1.7333698, -2.1984138, -1.7333698, -0.2565701, 0.2583165
9: -12.3942699, -11.8182421, -12.3942699, -11.8182421, -0.3340795, 0.3334663

Time for backsubstitution: 23.05 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.51 + 558.81 = 616.32 seconds
