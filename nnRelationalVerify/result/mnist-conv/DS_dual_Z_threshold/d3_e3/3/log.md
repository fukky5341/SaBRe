## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4083992475


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0849872, 1.0849872)
1: (-10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.3019185, 1.3019183)
2: (-10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2128665, 1.2128665)
3: (-4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.9018521, 0.9018519)
4: (-14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0938969, 1.0938969)
5: (8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6980789, 0.6980788)
6: (-4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0360487, 1.0360487)
7: (-15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2963200, 1.2963200)
8: (-0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.7521572, 0.7521572)
9: (-6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7734468, 0.7734468)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.81 + 34.95 = 57.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.4415111, upper bound: 0.4415114

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 976

Time for candidate selection: 0.47 seconds

### Candidate
type: DSZ, layer: 3, pos: 2216

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4358811, upper bound: 0.4361025
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4361010, upper bound: 0.4358826
time: 3.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.51
Output dim: 5, lower bound: -0.4358811, upper bound: 0.4361025
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.51
Output dim: 5, lower bound: -0.4361010, upper bound: 0.4358826

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0846467, 1.0845404
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.3007288, 1.2991769
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2105072, 1.2088957
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8951197, 0.8993120
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0925860, 1.0909638
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6962438, 0.6966611
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0357177, 1.0293753
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2963142, 1.2963128
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.7520928, 0.7522110
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7706614, 0.7705026

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 976

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4000787, upper bound: 0.4015942
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4000787, upper bound: 0.4015942
time: 2.84 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0849872, 1.0846462
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.2991772, 1.3019183
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2128665, 1.2105067
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8993120, 0.9018519
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0909638, 1.0938969
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6980789, 0.6962439
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0293758, 1.0360487
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2963123, 1.2963200
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.7521572, 0.7520928
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7734468, 0.7706614

Time for backsubstitution: 8.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 976

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4015927, upper bound: 0.4000802
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4015927, upper bound: 0.4000802
time: 2.79 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 13.84 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 13.84
Output dim: 5, lower bound: -0.4000787, upper bound: 0.4015942
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 13.84
Output dim: 5, lower bound: -0.4000787, upper bound: 0.4015942
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 13.84
Output dim: 5, lower bound: -0.4015927, upper bound: 0.4000802
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 13.84
Output dim: 5, lower bound: -0.4015927, upper bound: 0.4000802

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 57.77 + 34.94 = 92.70 seconds
