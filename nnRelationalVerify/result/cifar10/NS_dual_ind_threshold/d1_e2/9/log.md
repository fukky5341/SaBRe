## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 9)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0162870966


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7669811, -3.1572030, -3.7669811, -3.1572030, -0.1206218, 0.1206218)
1: (-2.7246246, -1.7280517, -2.7246246, -1.7280517, -0.2034255, 0.2034255)
2: (-0.7419281, -0.6119300, -0.7419281, -0.6119300, -0.0324391, 0.0324391)
3: (0.3925638, 0.4879531, 0.3925638, 0.4879531, -0.0413695, 0.0413695)
4: (-0.7734023, -0.6352448, -0.7734023, -0.6352448, -0.1042539, 0.1042539)
5: (0.0269803, 0.1058231, 0.0269803, 0.1058231, -0.0403319, 0.0403319)
6: (-0.7747244, -0.6061302, -0.7747244, -0.6061302, -0.0376193, 0.0376193)
7: (-0.4071949, -0.2342922, -0.4071949, -0.2342922, -0.0864264, 0.0864264)
8: (-3.1844516, -2.3247733, -3.1844516, -2.3247733, -0.1665304, 0.1665304)
9: (-0.7551889, 0.0517626, -0.7551889, 0.0517626, -0.2437801, 0.2437800)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.31 + 26.93 = 34.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0162988, upper bound: 0.0163024

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3233
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2910
type: A, layer: 1, pos: 2915
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3146
type: A, layer: 1, pos: 3369

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 387

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162990, upper bound: 0.0162773
time: 3.89 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0162983, upper bound: 0.0163039
time: 6.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.78 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.78
Output dim: 5, lower bound: -0.0162990, upper bound: 0.0162773
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.78
Output dim: 5, lower bound: -0.0162983, upper bound: 0.0163039

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.7669713, -3.1576884, -3.7669737, -3.1576064, -0.1202071, 0.1201555
1: -2.7246242, -1.7283130, -2.7246242, -1.7282629, -0.2032781, 0.2032487
2: -0.7419243, -0.6123995, -0.7419249, -0.6123129, -0.0320531, 0.0319719
3: 0.3925664, 0.4877939, 0.3925659, 0.4878173, -0.0412209, 0.0411880
4: -0.7733982, -0.6354625, -0.7733990, -0.6354213, -0.1040718, 0.1040311
5: 0.0269832, 0.1055789, 0.0269827, 0.1056131, -0.0400901, 0.0400579
6: -0.7747221, -0.6063771, -0.7747225, -0.6063304, -0.0374245, 0.0373902
7: -0.4068807, -0.2343013, -0.4069328, -0.2342996, -0.0860892, 0.0861403
8: -3.1844497, -2.3250935, -3.1844511, -2.3250337, -0.1663519, 0.1663167
9: -0.7550521, 0.0517621, -0.7550740, 0.0517612, -0.2433713, 0.2434409

Time for backsubstitution: 5.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2910
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3369

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3063

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162751, upper bound: 0.0162494
time: 6.12 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162753, upper bound: 0.0162527
time: 10.41 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.7675805, -3.1572609, -3.7669768, -3.1572506, -0.1211084, 0.1202188
1: -2.7249265, -1.7281446, -2.7246242, -1.7281280, -0.2034406, 0.2032757
2: -0.7425797, -0.6119321, -0.7419267, -0.6119322, -0.0330909, 0.0320298
3: 0.3923500, 0.4879537, 0.3925665, 0.4879369, -0.0416290, 0.0412377
4: -0.7736934, -0.6352250, -0.7734008, -0.6352458, -0.1045461, 0.1041892
5: 0.0265941, 0.1058214, 0.0269822, 0.1058215, -0.0405078, 0.0401622
6: -0.7751027, -0.6061279, -0.7747226, -0.6061313, -0.0379641, 0.0374284
7: -0.4071521, -0.2336604, -0.4071554, -0.2342923, -0.0861393, 0.0871594
8: -3.1848350, -2.3248644, -3.1844513, -2.3248475, -0.1667472, 0.1663512
9: -0.7548532, 0.0516753, -0.7548428, 0.0517616, -0.2434792, 0.2443148

Time for backsubstitution: 5.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3233
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2910
type: B, layer: 1, pos: 2915
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3146
type: B, layer: 1, pos: 3369

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3063

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162751, upper bound: 0.0162713
time: 40.26 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0162749, upper bound: 0.0162794
time: 18.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 64.16 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 64.16
Output dim: 5, lower bound: -0.0162751, upper bound: 0.0162494
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 64.16
Output dim: 5, lower bound: -0.0162753, upper bound: 0.0162527
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 64.16
Output dim: 5, lower bound: -0.0162751, upper bound: 0.0162713
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 64.16
Output dim: 5, lower bound: -0.0162749, upper bound: 0.0162794

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 34.24 + 97.15 = 131.39 seconds
