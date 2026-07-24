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
execution time: IAR + RelationalAnalysis = 21.38 + 35.20 = 56.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.4415111, upper bound: 0.4415114

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 186
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 186

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4353197, upper bound: 0.4353213
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4353197, upper bound: 0.4353213
time: 3.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.91
Output dim: 5, lower bound: -0.4353197, upper bound: 0.4353213
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.91
Output dim: 5, lower bound: -0.4353197, upper bound: 0.4353213

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0816135, 1.0821679
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1767764, 1.1916769
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2202468, 1.2295418
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.9010775, 0.8990916
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0958047, 1.0919361
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6897321, 0.6889056
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0358167, 1.0360587
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2679195, 1.2706773
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6926564, 0.7025297
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7736473, 0.7733948

Time for backsubstitution: 7.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1779

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4286800, upper bound: 0.4286815
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4286800, upper bound: 0.4286815
time: 3.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0821676, 1.0816135
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1916766, 1.1767759
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2295413, 1.2202470
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8990915, 0.9010776
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0919361, 1.0958047
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6889055, 0.6897321
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0360584, 1.0358167
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2706776, 1.2679198
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.7025295, 0.6926565
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7733948, 0.7736473

Time for backsubstitution: 7.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 677

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4241969, upper bound: 0.4311590
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4311471, upper bound: 0.4256534
time: 2.76 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 13.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.31
Output dim: 5, lower bound: -0.4286800, upper bound: 0.4286815
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.31
Output dim: 5, lower bound: -0.4286800, upper bound: 0.4286815
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 13.31
Output dim: 5, lower bound: -0.4241969, upper bound: 0.4311590
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 13.31
Output dim: 5, lower bound: -0.4311471, upper bound: 0.4256534

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0815873, 1.0822597
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1767697, 1.1916866
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2202382, 1.2295530
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.9010770, 0.8990670
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0953698, 1.0918276
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6897290, 0.6889223
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0357800, 1.0359645
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2675714, 1.2706306
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6926459, 0.7024685
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7738025, 0.7733853

Time for backsubstitution: 7.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 415

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4000038, upper bound: 0.4000052
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4000038, upper bound: 0.4000052
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0816135, 1.0821414
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1767764, 1.1916704
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2202468, 1.2295327
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.9010775, 0.8990911
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0956964, 1.0919361
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6897321, 0.6889025
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0358167, 1.0360222
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2678738, 1.2706773
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6926564, 0.7025191
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7736378, 0.7733948

Time for backsubstitution: 7.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2930

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 429

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4259908, upper bound: 0.4270861
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4270846, upper bound: 0.4259922
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0542860, 1.0536315
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1676981, 1.1578369
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1568422, 1.1527352
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8346679, 0.8389643
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9828839, 0.9706821
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6269617, 0.6309285
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0076160, 1.0127711
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2592373, 1.2553098
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6654447, 0.6636963
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6918874, 0.6866317

Time for backsubstitution: 7.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2930

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 918

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4234062, upper bound: 0.4309209
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4239565, upper bound: 0.4303703
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0541859, 1.0548041
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1727374, 1.1529167
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1620302, 1.1515498
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8404012, 0.8366539
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9668136, 0.9893029
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6347303, 0.6277883
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0130835, 1.0073745
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2580671, 1.2599890
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6741743, 0.6555715
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6863794, 0.6921788

Time for backsubstitution: 7.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2930

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 961

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4309635, upper bound: 0.4137628
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4192565, upper bound: 0.4254698
time: 2.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 19.88 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4000038, upper bound: 0.4000052
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4000038, upper bound: 0.4000052
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4259908, upper bound: 0.4270861
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4270846, upper bound: 0.4259922
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4234062, upper bound: 0.4309209
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4239565, upper bound: 0.4303703
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4309635, upper bound: 0.4137628
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.88
Output dim: 5, lower bound: -0.4192565, upper bound: 0.4254698

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0815840, 1.0820956
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1764288, 1.1910906
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2184110, 1.2275469
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8972957, 0.8967725
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0935488, 1.0887346
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6883610, 0.6879934
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0345926, 1.0342603
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2602777, 1.2608783
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6921831, 0.7018237
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7686751, 0.7700455

Time for backsubstitution: 7.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 192

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4215129, upper bound: 0.4236120
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4225390, upper bound: 0.4229004
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0815678, 1.0821118
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1761961, 1.1913233
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2182603, 1.2276971
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8987596, 0.8953093
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0924959, 1.0897877
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6888231, 0.6875316
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0340548, 1.0347979
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2580738, 1.2630813
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6919609, 0.7020460
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7702885, 0.7684324

Time for backsubstitution: 7.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3984084, upper bound: 0.3973160
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3984084, upper bound: 0.3973160
time: 2.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0542479, 1.0535717
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1675332, 1.1572678
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1564779, 1.1522384
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8347092, 0.8389133
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9828849, 0.9706767
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6266775, 0.6307435
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0075195, 1.0126879
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2591262, 1.2550240
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6653589, 0.6636314
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6918421, 0.6865730

Time for backsubstitution: 7.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1948

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2537

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4125278, upper bound: 0.4199031
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4125278, upper bound: 0.4199031
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0542259, 1.0535934
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1671288, 1.1576726
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1563454, 1.1523709
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8346169, 0.8390055
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9828787, 0.9706826
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6267769, 0.6306442
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0075333, 1.0126741
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2589512, 1.2551990
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6653799, 0.6636105
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6918283, 0.6865869

Time for backsubstitution: 7.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2853

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4237790, upper bound: 0.4288001
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4223864, upper bound: 0.4301927
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0492017, 1.0561571
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1709757, 1.1520066
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1622977, 1.1525908
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8407602, 0.8374411
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9558523, 0.9804888
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6396412, 0.6300383
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0170608, 1.0132513
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2575421, 1.2610738
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6745729, 0.6560695
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6853919, 0.6910774

Time for backsubstitution: 8.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1479

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4293357, upper bound: 0.4098363
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4270371, upper bound: 0.4121349
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0555389, 1.0498199
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1718273, 1.1511548
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1630712, 1.1518176
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8411884, 0.8370128
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9579995, 0.9783421
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6369804, 0.6326993
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0189605, 1.0113516
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2591519, 1.2594640
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6746726, 0.6559703
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6852779, 0.6911912

Time for backsubstitution: 7.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1446

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4126990, upper bound: 0.4250404
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4188001, upper bound: 0.4189612
time: 3.48 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.20 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4215129, upper bound: 0.4236120
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4225390, upper bound: 0.4229004
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.3984084, upper bound: 0.3973160
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.3984084, upper bound: 0.3973160
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4125278, upper bound: 0.4199031
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4125278, upper bound: 0.4199031
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4237790, upper bound: 0.4288001
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4223864, upper bound: 0.4301927
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4293357, upper bound: 0.4098363
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4270371, upper bound: 0.4121349
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4126990, upper bound: 0.4250404
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.20
Output dim: 5, lower bound: -0.4188001, upper bound: 0.4189612

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0724888, 1.0742040
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1726289, 1.1869721
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2183342, 1.2278318
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8933110, 0.8941432
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0752978, 1.0712028
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6841269, 0.6844809
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0355172, 1.0341449
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2583790, 1.2591450
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6922580, 0.7018985
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7716403, 0.7727342

Time for backsubstitution: 7.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3928367, upper bound: 0.3949358
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3928367, upper bound: 0.3949358
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0736942, 1.0730007
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1723104, 1.1872003
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.2189288, 1.2274702
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8949151, 0.8927876
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -1.0760159, 1.0710075
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6848483, 0.6832969
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0344772, 1.0354400
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2585449, 1.2589791
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6922575, 0.7018888
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.7713780, 0.7730105

Time for backsubstitution: 7.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 677
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2930

### Candidate
type: DSZ, layer: 3, pos: 1102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4026236, upper bound: 0.4028906
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4026236, upper bound: 0.4028906
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0543284, 1.0535007
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1683798, 1.1566494
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1535194, 1.1498775
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8354089, 0.8387495
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9831946, 0.9696977
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6265388, 0.6316721
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0021446, 0.9867415
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2589741, 1.2541060
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6602732, 0.6633799
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6918290, 0.6838115

Time for backsubstitution: 8.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2530

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4085467, upper bound: 0.4149899
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4076088, upper bound: 0.4159382
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0541768, 1.0535717
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1669154, 1.1572678
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1541173, 1.1522384
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8345454, 0.8389133
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9819052, 0.9706767
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6266775, 0.6306046
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0075195, 1.0073130
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2582088, 1.2550240
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6653589, 0.6585457
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6890810, 0.6865730

Time for backsubstitution: 7.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2853

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2629

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3856457, upper bound: 0.3917552
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3856457, upper bound: 0.3917552
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0501246, 1.0501204
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1610808, 1.1528969
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1569188, 1.1533604
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8369443, 0.8413515
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9778202, 0.9668834
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6258178, 0.6294110
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0059142, 1.0117798
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2541490, 1.2504644
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6633048, 0.6601431
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6921902, 0.6874336

Time for backsubstitution: 7.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 961

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4235953, upper bound: 0.4162137
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4117222, upper bound: 0.4286149
time: 7.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0507531, 1.0494919
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1623530, 1.1516252
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1573350, 1.1529441
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8369627, 0.8413329
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9790795, 0.9656243
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6255436, 0.6296852
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0066390, 1.0110548
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2542167, 1.2503967
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6619122, 0.6615355
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6926751, 0.6869485

Time for backsubstitution: 7.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 697

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1948

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4201814, upper bound: 0.4229470
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4150449, upper bound: 0.4280259
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0492706, 1.0548630
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1499705, 1.1314149
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1573827, 1.1499696
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8405938, 0.8373466
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9563200, 0.9807546
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6386559, 0.6280594
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0019979, 0.9980650
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2578635, 1.2624688
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6746510, 0.6565769
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6725445, 0.6773435

Time for backsubstitution: 8.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4006587, upper bound: 0.3811359
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4006587, upper bound: 0.3811359
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0479074, 1.0562263
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1503839, 1.1310015
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1596763, 1.1476755
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8406653, 0.8372747
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9561183, 0.9809566
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6376624, 0.6290528
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0018740, 0.9981892
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2589374, 1.2613950
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6750804, 0.6561475
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6716576, 0.6782302

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 976

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4263466, upper bound: 0.4049402
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4197432, upper bound: 0.4114425
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0418229, 1.0334103
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1630130, 1.1406960
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1556225, 1.1347160
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8397882, 0.8359518
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9221661, 0.9378471
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6183771, 0.6174715
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -0.9927170, 0.9951274
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2384353, 1.2345150
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6745567, 0.6557269
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6802955, 0.6861570

Time for backsubstitution: 7.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1689

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4015772, upper bound: 0.4199781
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4076502, upper bound: 0.4164480
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0391288, 1.0361040
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1613688, 1.1423404
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1459694, 1.1443686
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8401272, 0.8356128
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9175041, 0.9425089
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6217524, 0.6140960
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0027363, 0.9851077
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2342029, 1.2387474
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6744349, 0.6558545
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6802435, 0.6862090

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1948

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4167914, upper bound: 0.4116677
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4104816, upper bound: 0.4168336
time: 4.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 17.23 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.3928367, upper bound: 0.3949358
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.3928367, upper bound: 0.3949358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4026236, upper bound: 0.4028906
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4026236, upper bound: 0.4028906
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4085467, upper bound: 0.4149899
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4076088, upper bound: 0.4159382
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.3856457, upper bound: 0.3917552
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.3856457, upper bound: 0.3917552
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4235953, upper bound: 0.4162137
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4117222, upper bound: 0.4286149
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4201814, upper bound: 0.4229470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4150449, upper bound: 0.4280259
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4006587, upper bound: 0.3811359
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4006587, upper bound: 0.3811359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4263466, upper bound: 0.4049402
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4197432, upper bound: 0.4114425
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4015772, upper bound: 0.4199781
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4076502, upper bound: 0.4164480
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4167914, upper bound: 0.4116677
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.23
Output dim: 5, lower bound: -0.4104816, upper bound: 0.4168336

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0428915, 1.0466025
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1637309, 1.1527462
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1529717, 1.1451993
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8187029, 0.8179450
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9717109, 0.9565761
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6133127, 0.6217550
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -0.9574621, 0.9727449
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2486029, 1.2267714
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6025703, 0.6375028
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6793385, 0.6581801

Time for backsubstitution: 8.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 697

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 415

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3859939, upper bound: 0.3932356
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3859939, upper bound: 0.3932356
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0474300, 1.0420995
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1644766, 1.1520989
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1488414, 1.1495955
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8146043, 0.8221292
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9700730, 0.9582143
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6166215, 0.6184464
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -0.9886773, 0.9420590
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2316389, 1.2446527
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6344018, 0.6056769
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6661973, 0.6715536

Time for backsubstitution: 7.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 961

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4071875, upper bound: 0.4009573
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3924776, upper bound: 0.4155051
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0451295, 1.0514627
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1593194, 1.1519876
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1571860, 1.1544008
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8373032, 0.8421386
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9668598, 0.9580696
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6307342, 0.6316664
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0098906, 1.0176566
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2536249, 1.2515497
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6637034, 0.6606412
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6912022, 0.6863317

Time for backsubstitution: 7.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4212538, upper bound: 0.4156008
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4229143, upper bound: 0.4138031
time: 3.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0514667, 1.0451255
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1601715, 1.1511357
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1579595, 1.1536276
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8377314, 0.8417104
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9690070, 0.9559228
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6280732, 0.6343274
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0117908, 1.0157564
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2552342, 1.2499404
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6638029, 0.6605418
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6910884, 0.6864456

Time for backsubstitution: 7.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2914

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4084237, upper bound: 0.4234590
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4068289, upper bound: 0.4252242
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0376353, 1.0319850
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1465454, 1.1395531
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1414504, 1.1379318
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8202493, 0.8249716
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9087682, 0.9058793
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6177359, 0.6201621
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0044503, 1.0085046
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.1180973, 1.1266315
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6574073, 0.6568812
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6563506, 0.6524674

Time for backsubstitution: 7.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4177709, upper bound: 0.4222660
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195685, upper bound: 0.4206054
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0332460, 1.0363741
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1502814, 1.1358175
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1423230, 1.1370595
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8206010, 0.8246195
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9193373, 0.8953130
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6160207, 0.6218774
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0040889, 1.0088661
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.1304507, 1.1142769
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6572578, 0.6570306
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6581938, 0.6506240

Time for backsubstitution: 8.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1689

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4062740, upper bound: 0.4229090
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4099280, upper bound: 0.4192546
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0479076, 1.0562232
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1501555, 1.1308696
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1579247, 1.1464365
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8408091, 0.8374286
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9548197, 0.9804454
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6373236, 0.6285523
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0020738, 0.9982417
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2552571, 1.2587147
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6744955, 0.6553040
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6717243, 0.6783049

Time for backsubstitution: 7.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4166705, upper bound: 0.3985860
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4199042, upper bound: 0.3951966
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0479047, 1.0562260
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1502519, 1.1308558
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1584377, 1.1459243
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8408194, 0.8374184
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9556074, 0.9796581
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6371620, 0.6287142
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0019269, 0.9983892
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2562571, 1.2577147
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6744578, 0.6555626
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6717324, 0.6782968

Time for backsubstitution: 7.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1402

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 232

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4196951, upper bound: 0.4105850
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4168683, upper bound: 0.4113982
time: 3.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0337226, 1.0166593
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1618371, 1.1132085
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1458707, 1.1078920
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8270092, 0.7951105
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9179451, 0.9392967
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6180800, 0.6172824
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -0.9851649, 0.9802384
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2385688, 1.2327063
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6728420, 0.6534218
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6655240, 0.6868559

Time for backsubstitution: 7.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2530

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3970938, upper bound: 0.4149969
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3970691, upper bound: 0.4160604
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0250719, 1.0254495
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1355257, 1.1395197
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1287985, 1.1250777
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.7989469, 0.8231738
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9236152, 0.9336262
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6181880, 0.6171744
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -0.9778278, 0.9875760
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2366271, 1.2346480
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6722515, 0.6540122
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6809947, 0.6713856

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2375

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3781409, upper bound: 0.3913417
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3781409, upper bound: 0.3913417
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0239620, 1.0165479
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1450267, 1.1297338
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1300836, 1.1293552
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8236322, 0.8194699
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.8468635, 0.8828971
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6136317, 0.6042601
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0009041, 0.9829140
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.1006193, 1.1175189
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6699682, 0.6512387
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6438944, 0.6517029

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 976

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1922

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4122277, upper bound: 0.4104743
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4155966, upper bound: 0.4071053
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0195727, 1.0209370
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1487622, 1.1259980
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1309562, 1.1284826
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8238873, 0.8191179
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.8574326, 0.8718684
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6119165, 0.6059752
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0005426, 0.9832754
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.1129746, 1.1051643
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6698190, 0.6513879
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6457381, 0.6498597

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 918
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 725

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2537

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3984120, upper bound: 0.4061694
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3984017, upper bound: 0.4061694
time: 2.89 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.55 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3859939, upper bound: 0.3932356
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3859939, upper bound: 0.3932356
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4071875, upper bound: 0.4009573
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3924776, upper bound: 0.4155051
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4212538, upper bound: 0.4156008
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4229143, upper bound: 0.4138031
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4084237, upper bound: 0.4234590
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4068289, upper bound: 0.4252242
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4177709, upper bound: 0.4222660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4195685, upper bound: 0.4206054
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4062740, upper bound: 0.4229090
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4099280, upper bound: 0.4192546
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4166705, upper bound: 0.3985860
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4199042, upper bound: 0.3951966
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4196951, upper bound: 0.4105850
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4168683, upper bound: 0.4113982
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3970938, upper bound: 0.4149969
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3970691, upper bound: 0.4160604
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3781409, upper bound: 0.3913417
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3781409, upper bound: 0.3913417
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4122277, upper bound: 0.4104743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.4155966, upper bound: 0.4071053
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3984120, upper bound: 0.4061694
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.55
Output dim: 5, lower bound: -0.3984017, upper bound: 0.4061694

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0490732, 1.0374060
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1634209, 1.1501396
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1500754, 1.1498644
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8153839, 0.8224807
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9613428, 0.9473376
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6192119, 0.6236012
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -0.9947462, 0.9462287
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2328749, 1.2438047
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6344352, 0.6056112
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6648400, 0.6703099

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 550
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3744766, upper bound: 0.3959026
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3744766, upper bound: 0.3959026
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0394320, 1.0449986
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1556115, 1.1476135
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1567245, 1.1519589
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8372548, 0.8422310
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9572120, 0.9491744
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6306558, 0.6321999
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0060742, 1.0118060
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2451324, 1.2453811
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6603320, 0.6590202
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6910403, 0.6861813

Time for backsubstitution: 7.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2216
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 415

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3998043, upper bound: 0.3973550
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3998043, upper bound: 0.3973550
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.8452511, -10.9170074, -12.8452511, -10.9170074, -1.0386658, 1.0459628
1: -10.8974266, -8.8056717, -10.8974266, -8.8056717, -1.1549459, 1.1482797
2: -10.4901295, -8.9266987, -10.4901295, -8.9266987, -1.1547441, 1.1539392
3: -4.1869102, -2.6768126, -4.1869102, -2.6768126, -0.8373959, 0.8421452
4: -14.9127617, -12.9757938, -14.9127617, -12.9757938, -0.9573216, 0.9484217
5: 8.5341492, 9.5877390, 8.5341492, 9.5877390, -0.6312675, 0.6315879
6: -4.3723178, -2.6184142, -4.3723178, -2.6184142, -1.0040400, 1.0138578
7: -15.4907970, -13.4430943, -15.4907970, -13.4430943, -1.2474556, 1.2430570
8: -0.5657811, 0.6974521, -0.5657811, 0.6974521, -0.6620824, 0.6572697
9: -6.4858546, -5.2912550, -6.4858546, -5.2912550, -0.6910517, 0.6861697

Time for backsubstitution: 7.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1194
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2629
type: DSZ, layer: 3, pos: 232
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1948
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 415
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2805
type: DSZ, layer: 3, pos: 976
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2530
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1836
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1446
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 725
type: DSZ, layer: 3, pos: 1689
type: DSZ, layer: 3, pos: 2375
type: DSZ, layer: 3, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4209670, upper bound: 0.4056580
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4143333, upper bound: 0.4116659
time: 3.15 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 13.86 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 13.86
Output dim: 5, lower bound: -0.3744766, upper bound: 0.3959026
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 13.86
Output dim: 5, lower bound: -0.3744766, upper bound: 0.3959026
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 13.86
Output dim: 5, lower bound: -0.3998043, upper bound: 0.3973550
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 13.86
Output dim: 5, lower bound: -0.3998043, upper bound: 0.3973550
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 13.86
Output dim: 5, lower bound: -0.4209670, upper bound: 0.4056580
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 13.86
Output dim: 5, lower bound: -0.4143333, upper bound: 0.4116659
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4084237, upper bound: 0.4234590
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4068289, upper bound: 0.4252242
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4177709, upper bound: 0.4222660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4195685, upper bound: 0.4206054
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4062740, upper bound: 0.4229090
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4099280, upper bound: 0.4192546
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4166705, upper bound: 0.3985860
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4199042, upper bound: 0.3951966
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4196951, upper bound: 0.4105850
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4168683, upper bound: 0.4113982
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.3970938, upper bound: 0.4149969
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.3970691, upper bound: 0.4160604
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4122277, upper bound: 0.4104743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.86
Output dim: 5, lower bound: -0.4155966, upper bound: 0.4071053

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.58 + 547.61 = 604.19 seconds
