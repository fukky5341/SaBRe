## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.1823463684
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.8861728, 3.8861723)
1: (-10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.7987485, 2.7987485)
2: (-6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.8586493, 2.8586493)
3: (-2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.5133810, 2.5133805)
4: (-6.9938774, -2.8966291, -6.9938774, -2.8966291, -4.0211811, 4.0211816)
5: (-8.9602108, -5.7368851, -8.9602108, -5.7368851, -3.2127132, 3.2127137)
6: (-19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.8937092, 3.8937092)
7: (4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786)
8: (-7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.7680058, 2.7680058)
9: (-7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.4328909, 3.4328909)

## BASE Result
execution time: IAR + LP analysis = 15.39 + 33.32 = 48.71 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.29 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.7229785919189453
rel_dist={7: [-1.52484302938982, 1.5248425766176732]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.1847181417998263, 1.1847155369154763]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.7229785919189453
rel_dist={7: [-0.9114016965347282, 0.9113982906257574]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.7229785919189453
rel_dist={7: [-1.052759738359363, 1.0527567565617817]}

## Binary Search Result
Binary search time: 206.24 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3345.04 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265133, upper bound: 1.6158858
time: 4.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265133, upper bound: 1.6265126
time: 4.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.55
Output dim: 7, lower bound: -1.6265133, upper bound: 1.6158858
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.55
Output dim: 7, lower bound: -1.6265133, upper bound: 1.6265126

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.5972595, -13.5857925, -3.1234727, 3.1250405
1: -10.2623758, -7.4767718, -10.2654305, -7.4666820, -2.5149989, 2.5085196
2: -6.4378543, -3.5996532, -6.4559197, -3.5972705, -2.6264310, 2.6422110
3: -2.4340117, 0.1182419, -2.4377689, 0.1256915, -2.1372547, 2.1321177
4: -6.9883175, -2.9186773, -6.9938774, -2.8966291, -3.5357332, 3.5197964
5: -8.9537373, -5.7457619, -8.9602108, -5.7368851, -2.7727041, 2.7689753
6: -19.4427872, -15.5620022, -19.4462585, -15.5525494, -3.6878023, 3.6791468
7: 4.2643237, 6.9667125, 4.2598271, 6.9828057, -2.7184820, 2.7068853
8: -7.1617846, -4.4029832, -7.1687803, -4.4007745, -2.6804199, 2.6830809
9: -7.2016182, -3.7783484, -7.2100549, -3.7771640, -3.0454350, 3.0536122

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158878, upper bound: 1.6158856
time: 5.24 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6158878
time: 6.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.5972519, -13.5857944, -3.1403527, 3.1346211
1: -10.2822809, -7.4614153, -10.2654285, -7.4666867, -2.5357122, 2.5241065
2: -6.4601760, -3.5581908, -6.4559140, -3.5972736, -2.6490936, 2.6773546
3: -2.4422810, 0.1332530, -2.4377663, 0.1256859, -2.1500053, 2.1475527
4: -7.0440617, -2.8905511, -6.9938745, -2.8966470, -3.5716310, 3.5493207
5: -8.9876633, -5.7355204, -8.9602089, -5.7368917, -2.8107681, 2.7803917
6: -19.4601688, -15.5480824, -19.4462547, -15.5525570, -3.7134037, 3.6961298
7: 4.2270651, 6.9874487, 4.2598295, 6.9827995, -2.7557344, 2.7276192
8: -7.1751165, -4.3977704, -7.1687756, -4.4007754, -2.6979575, 2.6887491
9: -7.2168632, -3.7630327, -7.2100506, -3.7771640, -3.0617857, 3.0756216

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6265130
time: 5.09 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158878, upper bound: 1.6265151
time: 6.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.35 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 7, lower bound: -1.6158878, upper bound: 1.6158856
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6158878
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6265130
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 7, lower bound: -1.6158878, upper bound: 1.6265151

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.5882187, -13.5900822, -3.1161423, 3.1161427
1: -10.2623758, -7.4767718, -10.2623758, -7.4767718, -2.5046301, 2.5046301
2: -6.4378543, -3.5996532, -6.4378543, -3.5996532, -2.6239843, 2.6239843
3: -2.4340117, 0.1182419, -2.4340117, 0.1182419, -2.1277680, 2.1277683
4: -6.9883175, -2.9186773, -6.9883175, -2.9186773, -3.5131559, 3.5131559
5: -8.9537373, -5.7457619, -8.9537373, -5.7457619, -2.7625856, 2.7625856
6: -19.4427872, -15.5620022, -19.4427872, -15.5620022, -3.6741209, 3.6741204
7: 4.2643237, 6.9667125, 4.2643237, 6.9667125, -2.7023888, 2.7023888
8: -7.1617846, -4.4029832, -7.1617846, -4.4029832, -2.6743836, 2.6743839
9: -7.2016182, -3.7783484, -7.2016182, -3.7783484, -3.0418301, 3.0418301

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109282, upper bound: 1.6158834
time: 5.17 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158851, upper bound: 1.6158829
time: 5.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.6044312, -13.5805111, -3.1257300, 3.1329679
1: -10.2623758, -7.4767718, -10.2822809, -7.4614153, -2.5200343, 2.5253477
2: -6.4378543, -3.5996532, -6.4601760, -3.5581908, -2.6591010, 2.6463208
3: -2.4340117, 0.1182419, -2.4422810, 0.1332530, -2.1432061, 2.1359520
4: -6.9883175, -2.9186773, -7.0440617, -2.8905511, -3.5418911, 3.5490751
5: -8.9537373, -5.7457619, -8.9876633, -5.7355204, -2.7740059, 2.8002872
6: -19.4427872, -15.5620022, -19.4601688, -15.5480824, -3.6911068, 3.6915526
7: 4.2643237, 6.9667125, 4.2270651, 6.9874487, -2.7231250, 2.7396474
8: -7.1617846, -4.4029832, -7.1751165, -4.3977704, -2.6800556, 2.6884692
9: -7.2016182, -3.7783484, -7.2168632, -3.7630327, -3.0573654, 3.0581837

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109282, upper bound: 1.6158829
time: 6.06 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158831, upper bound: 1.6158852
time: 5.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.5882187, -13.5900822, -3.1329679, 3.1257305
1: -10.2822809, -7.4614153, -10.2623758, -7.4767718, -2.5253477, 2.5200343
2: -6.4601760, -3.5581908, -6.4378543, -3.5996532, -2.6463203, 2.6591010
3: -2.4422810, 0.1332530, -2.4340117, 0.1182419, -2.1359520, 2.1432061
4: -7.0440617, -2.8905511, -6.9883175, -2.9186773, -3.5490756, 3.5418916
5: -8.9876633, -5.7355204, -8.9537373, -5.7457619, -2.8002872, 2.7740064
6: -19.4601688, -15.5480824, -19.4427872, -15.5620022, -3.6915522, 3.6911073
7: 4.2270651, 6.9874487, 4.2643237, 6.9667125, -2.7396474, 2.7231250
8: -7.1751165, -4.3977704, -7.1617846, -4.4029832, -2.6884689, 2.6800559
9: -7.2168632, -3.7630327, -7.2016182, -3.7783484, -3.0581837, 3.0573645

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109260, upper bound: 1.6265099
time: 6.27 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158850, upper bound: 1.6265099
time: 5.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.6044312, -13.5805111, -3.1437654, 3.1437657
1: -10.2822809, -7.4614153, -10.2822809, -7.4614153, -2.5314302, 2.5314302
2: -6.4601760, -3.5581908, -6.4601760, -3.5581908, -2.6726789, 2.6726785
3: -2.4422810, 0.1332530, -2.4422810, 0.1332530, -2.1572137, 2.1572134
4: -7.0440617, -2.8905511, -7.0440617, -2.8905511, -3.5614433, 3.5614443
5: -8.9876633, -5.7355204, -8.9876633, -5.7355204, -2.8121786, 2.8121777
6: -19.4601688, -15.5480824, -19.4601688, -15.5480824, -3.7189484, 3.7189484
7: 4.2270651, 6.9874487, 4.2270651, 6.9874487, -2.7603836, 2.7603836
8: -7.1751165, -4.3977704, -7.1751165, -4.3977704, -2.7013674, 2.7013674
9: -7.2168632, -3.7630327, -7.2168632, -3.7630327, -3.0819702, 3.0819702

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109260, upper bound: 1.6265104
time: 5.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158829, upper bound: 1.6265126
time: 5.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6109282, upper bound: 1.6158834
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6158851, upper bound: 1.6158829
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6109282, upper bound: 1.6158829
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6158831, upper bound: 1.6158852
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6109260, upper bound: 1.6265099
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6158850, upper bound: 1.6265099
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6109260, upper bound: 1.6265104
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.90
Output dim: 7, lower bound: -1.6158829, upper bound: 1.6265126

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.5856018, -13.5917530, -3.1017079, 3.1086576
1: -10.2377262, -7.5479937, -10.2612934, -7.4954696, -2.4615231, 2.4324670
2: -6.3515334, -3.6288996, -6.4150143, -3.6006813, -2.5364065, 2.5693469
3: -2.4029136, 0.1065290, -2.4259477, 0.1171925, -2.0960646, 2.1077323
4: -6.9680395, -2.9741981, -6.9870405, -2.9330931, -3.4783421, 3.4561663
5: -8.9413986, -5.7605448, -8.9523335, -5.7495766, -2.7447720, 2.7461853
6: -19.4147167, -15.5752773, -19.4356461, -15.5628967, -3.6461449, 3.6458693
7: 4.2780347, 6.9527788, 4.2677011, 6.9649553, -2.6869206, 2.6850777
8: -7.1350555, -4.4596825, -7.1594329, -4.4180355, -2.6243758, 2.6152246
9: -7.1800356, -3.7988298, -7.1991644, -3.7837539, -2.9996409, 3.0184622

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5992339, upper bound: 1.6133643
time: 5.06 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109260, upper bound: 1.6158820
time: 5.85 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.5882187, -13.5900822, -3.1151376, 3.1161404
1: -10.2623730, -7.4767752, -10.2623758, -7.4767718, -2.5046291, 2.4637909
2: -6.4378519, -3.5996540, -6.4378543, -3.5996532, -2.5571938, 2.6239839
3: -2.4340096, 0.1182394, -2.4340117, 0.1182419, -2.1130047, 2.1277671
4: -6.9883184, -2.9186807, -6.9883175, -2.9186773, -3.5131550, 3.4799728
5: -8.9537363, -5.7457647, -8.9537373, -5.7457619, -2.7625847, 2.7531571
6: -19.4427834, -15.5620041, -19.4427872, -15.5620022, -3.6703873, 3.6710463
7: 4.2643232, 6.9667115, 4.2643237, 6.9667125, -2.7023892, 2.7023878
8: -7.1617851, -4.4029851, -7.1617846, -4.4029832, -2.6708493, 2.6400213
9: -7.2016177, -3.7783499, -7.2016182, -3.7783484, -3.0360231, 3.0557323

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158855, upper bound: 1.6109261
time: 5.26 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158855, upper bound: 1.6158829
time: 5.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.6018143, -13.5821791, -3.1112943, 3.1254849
1: -10.2377262, -7.5479937, -10.2812090, -7.4801044, -2.4769158, 2.4531922
2: -6.3515334, -3.6288996, -6.4373374, -3.5592113, -2.5714707, 2.5918427
3: -2.4029136, 0.1065290, -2.4342270, 0.1321986, -2.1114969, 2.1159236
4: -6.9680395, -2.9741981, -7.0427814, -2.9049563, -3.5070887, 3.4920921
5: -8.9413986, -5.7605448, -8.9862576, -5.7393370, -2.7561932, 2.7838874
6: -19.4147167, -15.5752773, -19.4530334, -15.5489788, -3.6631298, 3.6632962
7: 4.2780347, 6.9527788, 4.2304497, 6.9856830, -2.7076483, 2.7223291
8: -7.1350555, -4.4596825, -7.1727486, -4.4128251, -2.6300440, 2.6293027
9: -7.1800356, -3.7988298, -7.2143970, -3.7684388, -3.0151787, 3.0347962

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6099051, upper bound: 1.6133642
time: 5.12 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6216300, upper bound: 1.6158811
time: 5.50 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.6044312, -13.5805111, -3.1247268, 3.1329653
1: -10.2623730, -7.4767752, -10.2822809, -7.4614153, -2.5200334, 2.4845095
2: -6.4378519, -3.5996540, -6.4601760, -3.5581908, -2.5922742, 2.6463208
3: -2.4340096, 0.1182394, -2.4422810, 0.1332530, -2.1284432, 2.1359510
4: -6.9883184, -2.9186807, -7.0440617, -2.8905511, -3.5418901, 3.5158944
5: -8.9537363, -5.7457647, -8.9876633, -5.7355204, -2.7740049, 2.7908592
6: -19.4427834, -15.5620041, -19.4601688, -15.5480824, -3.6873446, 3.6884780
7: 4.2643232, 6.9667115, 4.2270651, 6.9874487, -2.7231255, 2.7396464
8: -7.1617851, -4.4029851, -7.1751165, -4.3977704, -2.6765218, 2.6540861
9: -7.2016177, -3.7783499, -7.2168632, -3.7630327, -3.0515575, 3.0719891

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265119, upper bound: 1.6109259
time: 5.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265101, upper bound: 1.6158831
time: 5.02 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.5856018, -13.5917530, -3.1185365, 3.1182404
1: -10.2576618, -7.5326185, -10.2612934, -7.4954696, -2.4780822, 2.4478645
2: -6.3738647, -3.5874131, -6.4150143, -3.6006813, -2.5587502, 2.5783052
3: -2.4112453, 0.1215175, -2.4259477, 0.1171925, -2.1042848, 2.1231482
4: -7.0237694, -2.9460731, -6.9870405, -2.9330931, -3.4977345, 3.4849176
5: -8.9753370, -5.7503128, -8.9523335, -5.7495766, -2.7824888, 2.7576013
6: -19.4321251, -15.5613813, -19.4356461, -15.5628967, -3.6636238, 3.6628437
7: 4.2407608, 6.9734745, 4.2677011, 6.9649553, -2.7241945, 2.7057734
8: -7.1483364, -4.4544744, -7.1594329, -4.4180355, -2.6384416, 2.6208842
9: -7.1952000, -3.7835121, -7.1991644, -3.7837539, -3.0159049, 3.0339975

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5992316, upper bound: 1.6240021
time: 5.09 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109237, upper bound: 1.6265084
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5882187, -13.5900822, -3.1319633, 3.1257291
1: -10.2822819, -7.4614186, -10.2623758, -7.4767718, -2.5229967, 2.4791956
2: -6.4601712, -3.5581923, -6.4378543, -3.5996532, -2.5795298, 2.6346455
3: -2.4422777, 0.1332527, -2.4340117, 0.1182419, -2.1211882, 2.1432056
4: -7.0440588, -2.8905525, -6.9883175, -2.9186773, -3.5339341, 3.5087080
5: -8.9876623, -5.7355213, -8.9537373, -5.7457619, -2.8002872, 2.7645755
6: -19.4601593, -15.5480824, -19.4427872, -15.5620022, -3.6878214, 3.6880422
7: 4.2270684, 6.9874468, 4.2643237, 6.9667125, -2.7396441, 2.7231231
8: -7.1751165, -4.3977718, -7.1617846, -4.4029832, -2.6849394, 2.6457019
9: -7.2168646, -3.7630339, -7.2016182, -3.7783484, -3.0524006, 3.0712686

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158852, upper bound: 1.6216323
time: 8.37 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158852, upper bound: 1.6265092
time: 6.84 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.6018143, -13.5821791, -3.1293340, 3.1362774
1: -10.2576618, -7.5326185, -10.2812090, -7.4801044, -2.4883366, 2.4592676
2: -6.3738647, -3.5874131, -6.4373374, -3.5592113, -2.5851116, 2.6008010
3: -2.4112453, 0.1215175, -2.4342270, 0.1321986, -2.1255398, 2.1371624
4: -7.0237694, -2.9460731, -7.0427814, -2.9049563, -3.5264292, 3.5044818
5: -8.9753370, -5.7503128, -8.9862576, -5.7393370, -2.7943792, 2.7957726
6: -19.4321251, -15.5613813, -19.4530334, -15.5489788, -3.6910191, 3.6906800
7: 4.2407608, 6.9734745, 4.2304497, 6.9856830, -2.7449222, 2.7430248
8: -7.1483364, -4.4544744, -7.1727486, -4.4128251, -2.6514292, 2.6422193
9: -7.1952000, -3.7835121, -7.2143970, -3.7684388, -3.0396962, 3.0585828

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6000543, upper bound: 1.6240029
time: 5.32 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6117790, upper bound: 1.6265090
time: 5.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.6044312, -13.5805111, -3.1427622, 3.1437638
1: -10.2822819, -7.4614186, -10.2822809, -7.4614153, -2.5314293, 2.4905906
2: -6.4601712, -3.5581923, -6.4601760, -3.5581908, -2.6058884, 2.6571293
3: -2.4422777, 0.1332527, -2.4422810, 0.1332530, -2.1424503, 2.1572130
4: -7.0440588, -2.8905525, -7.0440617, -2.8905511, -3.5614424, 3.5282593
5: -8.9876623, -5.7355213, -8.9876633, -5.7355204, -2.8121767, 2.8027482
6: -19.4601593, -15.5480824, -19.4601688, -15.5480824, -3.7151928, 3.7158852
7: 4.2270684, 6.9874468, 4.2270651, 6.9874487, -2.7603803, 2.7603817
8: -7.1751165, -4.3977718, -7.1751165, -4.3977704, -2.6978383, 2.6669929
9: -7.2168646, -3.7630339, -7.2168632, -3.7630327, -3.0761871, 3.0957785

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166362, upper bound: 1.6216318
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166344, upper bound: 1.6265107
time: 5.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.98 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.5992339, upper bound: 1.6133643
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6109260, upper bound: 1.6158820
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6158855, upper bound: 1.6109261
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6158855, upper bound: 1.6158829
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6099051, upper bound: 1.6133642
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6216300, upper bound: 1.6158811
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6265119, upper bound: 1.6109259
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6265101, upper bound: 1.6158831
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.5992316, upper bound: 1.6240021
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6109237, upper bound: 1.6265084
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6158852, upper bound: 1.6216323
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6158852, upper bound: 1.6265092
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6000543, upper bound: 1.6240029
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6117790, upper bound: 1.6265090
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6166362, upper bound: 1.6216318
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 7, lower bound: -1.6166344, upper bound: 1.6265107

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.5745716, -13.5985107, -17.5761566, -13.6135349, -3.0781984, 3.0926538
1: -10.2360249, -7.5487485, -10.2505178, -7.5008335, -2.4543943, 2.4203095
2: -6.3468728, -3.6294789, -6.3862920, -3.6079888, -2.5238285, 2.5380995
3: -2.3945169, 0.1057823, -2.3745484, 0.1066229, -2.0694237, 2.0555637
4: -6.9673076, -2.9779592, -6.9803143, -2.9563215, -3.4545259, 3.4464283
5: -8.9326591, -5.7608948, -8.8985386, -5.7569742, -2.7291088, 2.6928720
6: -19.4138451, -15.5827522, -19.4262581, -15.6087303, -3.5996656, 3.6298122
7: 4.2789607, 6.9467688, 4.2764544, 6.9281015, -2.6491408, 2.6703143
8: -7.1336780, -4.4643803, -7.1473355, -4.4468279, -2.5939794, 2.5967696
9: -7.1791215, -3.8011398, -7.1922774, -3.7981288, -2.9840956, 3.0080371

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5992339, upper bound: 1.6040845
time: 5.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5992320, upper bound: 1.6133645
time: 4.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.5856018, -13.5917549, -3.0959854, 3.1078739
1: -10.2377262, -7.5479937, -10.2612944, -7.4954696, -2.4645171, 2.4324656
2: -6.3515334, -3.6288996, -6.4150119, -3.6006804, -2.5364065, 2.5579643
3: -2.4029136, 0.1065290, -2.4259427, 0.1171930, -2.0960636, 2.0733316
4: -6.9680395, -2.9741981, -6.9870410, -2.9330945, -3.4721394, 3.4561658
5: -8.9413986, -5.7605448, -8.9523296, -5.7495766, -2.7447720, 2.7105269
6: -19.4147167, -15.5752773, -19.4356480, -15.5628996, -3.6286421, 3.6458683
7: 4.2780347, 6.9527788, 4.2677031, 6.9649510, -2.6869164, 2.6850758
8: -7.1350555, -4.4596825, -7.1594305, -4.4180388, -2.6083589, 2.6148698
9: -7.1800356, -3.7988298, -7.1991653, -3.7837555, -2.9953628, 3.0184617

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085225, upper bound: 1.6040848
time: 5.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085245, upper bound: 1.6158815
time: 5.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.5882149, -13.5900860, -17.5757713, -13.5949659, -3.1114159, 3.1035225
1: -10.2623749, -7.4767752, -10.2377262, -7.5479937, -2.4334807, 2.4719229
2: -6.4378529, -3.5996537, -6.3515334, -3.6288996, -2.5707965, 2.5373507
3: -2.4340079, 0.1182394, -2.4029136, 0.1065290, -2.1157751, 2.0972333
4: -6.9883175, -2.9186797, -6.9680395, -2.9741981, -3.4574070, 3.4841356
5: -8.9537354, -5.7457647, -8.9413986, -5.7605448, -2.7475591, 2.7494273
6: -19.4427814, -15.5620031, -19.4147167, -15.5752773, -3.6571426, 3.6442313
7: 4.2643232, 6.9667120, 4.2780347, 6.9527788, -2.6884556, 2.6886773
8: -7.1617823, -4.4029865, -7.1350555, -4.4596825, -2.6144648, 2.6358705
9: -7.2016172, -3.7783501, -7.1800356, -3.7988298, -3.0160589, 3.0134215

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133646, upper bound: 1.5992315
time: 5.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158832, upper bound: 1.6109240
time: 5.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.5882168, -13.5900850, -3.1151366, 3.1151359
1: -10.2623730, -7.4767752, -10.2623730, -7.4767752, -2.4637899, 2.4637909
2: -6.4378519, -3.5996540, -6.4378519, -3.5996540, -2.5571923, 2.5571928
3: -2.4340096, 0.1182394, -2.4340096, 0.1182394, -2.1130042, 2.1130040
4: -6.9883184, -2.9186807, -6.9883184, -2.9186807, -3.4799709, 3.4799709
5: -8.9537363, -5.7457647, -8.9537363, -5.7457647, -2.7531567, 2.7531567
6: -19.4427834, -15.5620041, -19.4427834, -15.5620041, -3.6703815, 3.6703820
7: 4.2643232, 6.9667115, 4.2643232, 6.9667115, -2.7023883, 2.7023883
8: -7.1617851, -4.4029851, -7.1617851, -4.4029851, -2.6400185, 2.6400180
9: -7.2016177, -3.7783499, -7.2016177, -3.7783499, -3.0557280, 3.0557280

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133649, upper bound: 1.5992315
time: 4.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158835, upper bound: 1.6109240
time: 5.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.5745716, -13.5985107, -17.5923100, -13.6039715, -3.0877714, 3.1094284
1: -10.2360249, -7.5487485, -10.2704601, -7.4855156, -2.4697390, 2.4410801
2: -6.3468728, -3.6294789, -6.4086056, -3.5665157, -2.5460334, 2.5605907
3: -2.3945169, 0.1057823, -2.3828931, 0.1216087, -2.0848632, 2.0638416
4: -6.9673076, -2.9779592, -7.0360651, -2.9282517, -3.4832277, 3.4728608
5: -8.9326591, -5.7608948, -8.9325600, -5.7467403, -2.7405229, 2.7306714
6: -19.4138451, -15.5827522, -19.4436722, -15.5948629, -3.6165972, 3.6473222
7: 4.2789607, 6.9467688, 4.2391529, 6.9487762, -2.6698155, 2.7076159
8: -7.1336780, -4.4643803, -7.1606021, -4.4416242, -2.5996366, 2.6108043
9: -7.1791215, -3.8011398, -7.2074509, -3.7827809, -2.9996614, 3.0243034

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6099032, upper bound: 1.6040847
time: 5.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6099051, upper bound: 1.6133642
time: 5.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.6018105, -13.5821810, -3.1055679, 3.1247015
1: -10.2377262, -7.5479937, -10.2812071, -7.4801035, -2.4799113, 2.4531918
2: -6.3515334, -3.6288996, -6.4373341, -3.5592129, -2.5684838, 2.5804596
3: -2.4029136, 0.1065290, -2.4342217, 0.1321985, -2.1114969, 2.0815232
4: -6.9680395, -2.9741981, -7.0427828, -2.9049582, -3.5008879, 3.4904099
5: -8.9413986, -5.7605448, -8.9862537, -5.7393360, -2.7561927, 2.7482266
6: -19.4147167, -15.5752773, -19.4530296, -15.5489807, -3.6456280, 3.6632967
7: 4.2780347, 6.9527788, 4.2304497, 6.9856787, -2.7076440, 2.7223291
8: -7.1350555, -4.4596825, -7.1727490, -4.4128275, -2.6140285, 2.6289515
9: -7.1800356, -3.7988298, -7.2143946, -3.7684402, -3.0108991, 3.0347953

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6191898, upper bound: 1.6040849
time: 4.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6191899, upper bound: 1.6158833
time: 4.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.5882149, -13.5900860, -17.5919857, -13.5853930, -3.1209993, 3.1203508
1: -10.2623749, -7.4767752, -10.2576618, -7.5326185, -2.4488788, 2.4803703
2: -6.4378529, -3.5996537, -6.3738647, -3.5874131, -2.5797544, 2.5596933
3: -2.4340079, 0.1182394, -2.4112453, 0.1215175, -2.1311913, 2.1054535
4: -6.9883175, -2.9186797, -7.0237694, -2.9460731, -3.4861584, 3.4996557
5: -8.9537354, -5.7457647, -8.9753370, -5.7503128, -2.7589760, 2.7871437
6: -19.4427814, -15.5620031, -19.4321251, -15.5613813, -3.6741171, 3.6617093
7: 4.2643232, 6.9667120, 4.2407608, 6.9734745, -2.7091513, 2.7259512
8: -7.1617823, -4.4029865, -7.1483364, -4.4544744, -2.6201239, 2.6498497
9: -7.2016172, -3.7783501, -7.1952000, -3.7835121, -3.0315948, 3.0296865

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6240042, upper bound: 1.5992309
time: 5.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265098, upper bound: 1.6109235
time: 5.48 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.6044273, -13.5805111, -3.1247263, 3.1319613
1: -10.2623730, -7.4767752, -10.2822819, -7.4614186, -2.4791946, 2.4845090
2: -6.4378519, -3.5996540, -6.4601712, -3.5581923, -2.5886707, 2.5795293
3: -2.4340096, 0.1182394, -2.4422777, 0.1332527, -2.1284423, 2.1211879
4: -6.9883184, -2.9186807, -7.0440588, -2.8905525, -3.5087061, 3.5111060
5: -8.9537363, -5.7457647, -8.9876623, -5.7355213, -2.7645755, 2.7908587
6: -19.4427834, -15.5620041, -19.4601593, -15.5480824, -3.6873417, 3.6878176
7: 4.2643232, 6.9667115, 4.2270684, 6.9874468, -2.7231236, 2.7396431
8: -7.1617851, -4.4029851, -7.1751165, -4.3977718, -2.6456990, 2.6540837
9: -7.2016177, -3.7783499, -7.2168646, -3.7630339, -3.0712643, 3.0719872

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6240026, upper bound: 1.5992308
time: 5.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265083, upper bound: 1.6109234
time: 5.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.5907784, -13.5889359, -17.5761566, -13.6135349, -3.0950174, 3.1022341
1: -10.2559643, -7.5333819, -10.2505178, -7.5008335, -2.4689779, 2.4356995
2: -6.3692036, -3.5879893, -6.3862920, -3.6079888, -2.5461721, 2.5470538
3: -2.4028568, 0.1207665, -2.3745484, 0.1066229, -2.0760193, 2.0709772
4: -7.0230465, -2.9498436, -6.9803143, -2.9563215, -3.4738626, 3.4751682
5: -8.9666119, -5.7506628, -8.8985386, -5.7569742, -2.7510691, 2.7042861
6: -19.4312611, -15.5688639, -19.4262581, -15.6087303, -3.6171732, 3.6467752
7: 4.2416792, 6.9674573, 4.2764544, 6.9281015, -2.6864223, 2.6910028
8: -7.1469526, -4.4591761, -7.1473355, -4.4468279, -2.6080389, 2.6024282
9: -7.1942716, -3.7858160, -7.1922774, -3.7981288, -3.0003481, 3.0235772

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5992316, upper bound: 1.6147225
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5992334, upper bound: 1.6240019
time: 5.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.5856018, -13.5917549, -3.1128092, 3.1174569
1: -10.2576618, -7.5326185, -10.2612944, -7.4954696, -2.4804678, 2.4478636
2: -6.3738647, -3.5874131, -6.4150119, -3.6006804, -2.5587502, 2.5669227
3: -2.4112453, 0.1215175, -2.4259427, 0.1171930, -2.1042833, 2.0887480
4: -7.0237694, -2.9460731, -6.9870410, -2.9330945, -3.4915323, 3.4849167
5: -8.9753370, -5.7503128, -8.9523296, -5.7495766, -2.7824879, 2.7219424
6: -19.4321251, -15.5613813, -19.4356480, -15.5628996, -3.6461210, 3.6628423
7: 4.2407608, 6.9734745, 4.2677031, 6.9649510, -2.7241902, 2.7057714
8: -7.1483364, -4.4544744, -7.1594305, -4.4180388, -2.6224098, 2.6205292
9: -7.1952000, -3.7835121, -7.1991653, -3.7837555, -3.0116286, 3.0339971

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085222, upper bound: 1.6147228
time: 5.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085222, upper bound: 1.6265080
time: 4.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5757713, -13.5949659, -3.1282415, 3.1131113
1: -10.2822800, -7.4614191, -10.2377262, -7.5479937, -2.4518049, 2.4874740
2: -6.4601707, -3.5581927, -6.3515334, -3.6288996, -2.5932984, 2.5479603
3: -2.4422772, 0.1332521, -2.4029136, 0.1065290, -2.1235564, 2.1126716
4: -7.0440578, -2.8905535, -6.9680395, -2.9741981, -3.4781833, 3.5128388
5: -8.9876623, -5.7355223, -8.9413986, -5.7605448, -2.7852616, 2.7608495
6: -19.4601631, -15.5480824, -19.4147167, -15.5752773, -3.6745739, 3.6612272
7: 4.2270689, 6.9874468, 4.2780347, 6.9527788, -2.7257099, 2.7094121
8: -7.1751156, -4.3977718, -7.1350555, -4.4596825, -2.6285563, 2.6415229
9: -7.2168636, -3.7630348, -7.1800356, -3.7988298, -3.0324364, 3.0289569

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133643, upper bound: 1.6099022
time: 5.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158829, upper bound: 1.6216295
time: 7.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5882168, -13.5900850, -3.1319613, 3.1247258
1: -10.2822819, -7.4614186, -10.2623730, -7.4767752, -2.4845085, 2.4791946
2: -6.4601712, -3.5581923, -6.4378519, -3.5996540, -2.5795293, 2.5886710
3: -2.4422777, 0.1332527, -2.4340096, 0.1182394, -2.1211877, 2.1284420
4: -7.0440588, -2.8905525, -6.9883184, -2.9186807, -3.5111060, 3.5087061
5: -8.9876623, -5.7355213, -8.9537363, -5.7457647, -2.7908583, 2.7645750
6: -19.4601593, -15.5480824, -19.4427834, -15.5620041, -3.6878176, 3.6873417
7: 4.2270684, 6.9874468, 4.2643232, 6.9667115, -2.7396431, 2.7231236
8: -7.1751165, -4.3977718, -7.1617851, -4.4029851, -2.6540837, 2.6456990
9: -7.2168646, -3.7630339, -7.2016177, -3.7783499, -3.0719872, 3.0712638

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133664, upper bound: 1.6099033
time: 5.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158832, upper bound: 1.6099026
time: 7.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.5907784, -13.5889359, -17.5923100, -13.6039715, -3.1058016, 3.1202197
1: -10.2559643, -7.5333819, -10.2704601, -7.4855156, -2.4811668, 2.4471483
2: -6.3692036, -3.5879893, -6.4086056, -3.5665157, -2.5685306, 2.5695455
3: -2.4028568, 0.1207665, -2.3828931, 0.1216087, -2.0939660, 2.0850775
4: -7.0230465, -2.9498436, -7.0360651, -2.9282517, -3.5024920, 3.4947510
5: -8.9666119, -5.7506628, -8.9325600, -5.7467403, -2.7627010, 2.7425561
6: -19.4312611, -15.5688639, -19.4436722, -15.5948629, -3.6445131, 3.6733327
7: 4.2416792, 6.9674573, 4.2391529, 6.9487762, -2.7070971, 2.7283044
8: -7.1469526, -4.4591761, -7.1606021, -4.4416242, -2.6210313, 2.6238151
9: -7.1942716, -3.7858160, -7.2074509, -3.7827809, -3.0241680, 3.0480957

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6000543, upper bound: 1.6147227
time: 5.23 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6000524, upper bound: 1.6240030
time: 4.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.6018105, -13.5821810, -3.1236010, 3.1354935
1: -10.2576618, -7.5326185, -10.2812071, -7.4801035, -2.4913321, 2.4592662
2: -6.3738647, -3.5874131, -6.4373341, -3.5592129, -2.5851102, 2.5894175
3: -2.4112453, 0.1215175, -2.4342217, 0.1321985, -2.1255398, 2.1027627
4: -7.0237694, -2.9460731, -7.0427828, -2.9049582, -3.5202289, 3.5044823
5: -8.9753370, -5.7503128, -8.9862537, -5.7393360, -2.7943788, 2.7601123
6: -19.4321251, -15.5613813, -19.4530296, -15.5489807, -3.6735172, 3.6906791
7: 4.2407608, 6.9734745, 4.2304497, 6.9856787, -2.7449179, 2.7430248
8: -7.1483364, -4.4544744, -7.1727490, -4.4128275, -2.6353979, 2.6418681
9: -7.1952000, -3.7835121, -7.2143946, -3.7684402, -3.0354166, 3.0585823

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6093396, upper bound: 1.6147226
time: 4.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6093396, upper bound: 1.6265088
time: 5.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5919857, -13.5853930, -3.1390333, 3.1311502
1: -10.2822800, -7.4614191, -10.2576618, -7.5326185, -2.4602747, 2.4959211
2: -6.4601707, -3.5581927, -6.3738647, -3.5874131, -2.6022568, 2.5704513
3: -2.4422772, 0.1332521, -2.4112453, 0.1215175, -2.1415000, 2.1267145
4: -7.0440578, -2.8905535, -7.0237694, -2.9460731, -3.5057087, 3.5283585
5: -8.9876623, -5.7355223, -8.9753370, -5.7503128, -2.7971478, 2.7990341
6: -19.4601631, -15.5480824, -19.4321251, -15.5613813, -3.7019567, 3.6891150
7: 4.2270689, 6.9874468, 4.2407608, 6.9734745, -2.7464056, 2.7466860
8: -7.1751156, -4.3977718, -7.1483364, -4.4544744, -2.6414437, 2.6602190
9: -7.2168636, -3.7630348, -7.1952000, -3.7835121, -3.0562239, 3.0534739

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141439, upper bound: 1.6099025
time: 5.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166323, upper bound: 1.6216306
time: 5.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.99 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.5992339, upper bound: 1.6040845
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.5992320, upper bound: 1.6133645
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6085225, upper bound: 1.6040848
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6085245, upper bound: 1.6158815
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6133646, upper bound: 1.5992315
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6158832, upper bound: 1.6109240
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6133649, upper bound: 1.5992315
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6158835, upper bound: 1.6109240
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6099032, upper bound: 1.6040847
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6099051, upper bound: 1.6133642
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6191898, upper bound: 1.6040849
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6191899, upper bound: 1.6158833
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6240042, upper bound: 1.5992309
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6265098, upper bound: 1.6109235
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6240026, upper bound: 1.5992308
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6265083, upper bound: 1.6109234
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.5992316, upper bound: 1.6147225
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.5992334, upper bound: 1.6240019
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6085222, upper bound: 1.6147228
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6085222, upper bound: 1.6265080
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6133643, upper bound: 1.6099022
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6158829, upper bound: 1.6216295
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6133664, upper bound: 1.6099033
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6158832, upper bound: 1.6099026
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6000543, upper bound: 1.6147227
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6000524, upper bound: 1.6240030
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6093396, upper bound: 1.6147226
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6093396, upper bound: 1.6265088
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6141439, upper bound: 1.6099025
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.99
Output dim: 7, lower bound: -1.6166323, upper bound: 1.6216306
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.99
Output dim: 7, lower bound: -1.6166344, upper bound: 1.6265107
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.7229785919189453
rel_dist={7: [-1.6265317094795417, 1.626531505248428]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100426, upper bound: 1.3032400
time: 5.05 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105111, upper bound: 1.3105086
time: 5.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.57
Output dim: 7, lower bound: -1.3100426, upper bound: 1.3032400
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.57
Output dim: 7, lower bound: -1.3105111, upper bound: 1.3105086

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.5943260, -13.5872059, -2.6688042, 2.6698627
1: -10.2623758, -7.4767718, -10.2644424, -7.4699593, -2.3157997, 2.3114333
2: -6.4378543, -3.5996532, -6.4500575, -3.5980411, -2.4210196, 2.4316754
3: -2.4340117, 0.1182419, -2.4365590, 0.1232623, -1.9110980, 1.9076295
4: -6.9883175, -2.9186773, -6.9921036, -2.9037902, -3.2411156, 3.2304001
5: -8.9537373, -5.7457619, -8.9581261, -5.7397661, -2.5092487, 2.5067415
6: -19.4427872, -15.5620022, -19.4451408, -15.5556211, -3.3085718, 3.3027296
7: 4.2643237, 6.9667125, 4.2612715, 6.9775810, -2.7132573, 2.7054410
8: -7.1617846, -4.4029832, -7.1664982, -4.4014907, -2.4548159, 2.4566312
9: -7.2016182, -3.7783484, -7.2073135, -3.7775440, -2.7649565, 2.7704659

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032399, upper bound: 1.3032397
time: 5.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032399, upper bound: 1.3032393
time: 5.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.5972519, -13.5858002, -2.6875515, 2.6823378
1: -10.2822809, -7.4614153, -10.2654266, -7.4666901, -2.3398790, 2.3266315
2: -6.4601760, -3.5581908, -6.4559107, -3.5972733, -2.4416447, 2.4661918
3: -2.4422810, 0.1332530, -2.4377654, 0.1256843, -1.9263749, 1.9244847
4: -7.0440617, -2.8905511, -6.9938736, -2.8966587, -3.2829666, 3.2552896
5: -8.9876633, -5.7355204, -8.9602051, -5.7368975, -2.5505490, 2.5202169
6: -19.4601688, -15.5480824, -19.4462528, -15.5525637, -3.3375998, 3.3213367
7: 4.2270651, 6.9874487, 4.2598314, 6.9827957, -2.7557306, 2.7276173
8: -7.1751165, -4.3977704, -7.1687756, -4.4007754, -2.4726090, 2.4651015
9: -7.2168632, -3.7630327, -7.2100439, -3.7771654, -2.7824669, 2.7955036

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032421, upper bound: 1.3100408
time: 5.26 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032421, upper bound: 1.3105088
time: 5.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.69
Output dim: 7, lower bound: -1.3032399, upper bound: 1.3032397
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.69
Output dim: 7, lower bound: -1.3032399, upper bound: 1.3032393
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.69
Output dim: 7, lower bound: -1.3032421, upper bound: 1.3100408
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.69
Output dim: 7, lower bound: -1.3032421, upper bound: 1.3105088

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.5882187, -13.5900822, -2.6638622, 2.6638618
1: -10.2623758, -7.4767718, -10.2623758, -7.4767718, -2.3087997, 2.3087997
2: -6.4378543, -3.5996532, -6.4378543, -3.5996532, -2.4193630, 2.4193635
3: -2.4340117, 0.1182419, -2.4340117, 0.1182419, -1.9047027, 1.9047024
4: -6.9883175, -2.9186773, -6.9883175, -2.9186773, -3.2258711, 3.2258711
5: -8.9537373, -5.7457619, -8.9537373, -5.7457619, -2.5024128, 2.5024133
6: -19.4427872, -15.5620022, -19.4427872, -15.5620022, -3.2993307, 3.2993307
7: 4.2643237, 6.9667125, 4.2643237, 6.9667125, -2.7023888, 2.7023888
8: -7.1617846, -4.4029832, -7.1617846, -4.4029832, -2.4507394, 2.4507396
9: -7.2016182, -3.7783484, -7.2016182, -3.7783484, -2.7625141, 2.7625141

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003824, upper bound: 1.3032384
time: 5.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032385, upper bound: 1.3032382
time: 5.03 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.6044312, -13.5805111, -2.6734500, 2.6806870
1: -10.2623758, -7.4767718, -10.2822809, -7.4614153, -2.3242040, 2.3295174
2: -6.4378543, -3.5996532, -6.4601760, -3.5581908, -2.4479403, 2.4417000
3: -2.4340117, 0.1182419, -2.4422810, 0.1332530, -1.9201403, 1.9128861
4: -6.9883175, -2.9186773, -7.0440617, -2.8905511, -3.2546062, 3.2604127
5: -8.9537373, -5.7457619, -8.9876633, -5.7355204, -2.5138340, 2.5401154
6: -19.4427872, -15.5620022, -19.4601688, -15.5480824, -3.3163166, 3.3167629
7: 4.2643237, 6.9667125, 4.2270651, 6.9874487, -2.7231250, 2.7396474
8: -7.1617846, -4.4029832, -7.1751165, -4.3977704, -2.4564114, 2.4648249
9: -7.2016182, -3.7783484, -7.2168632, -3.7630327, -2.7780495, 2.7788677

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003824, upper bound: 1.3032382
time: 6.04 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032385, upper bound: 1.3032381
time: 6.23 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.5882187, -13.5900822, -2.6806870, 2.6734495
1: -10.2822809, -7.4614153, -10.2623758, -7.4767718, -2.3295183, 2.3242040
2: -6.4601760, -3.5581908, -6.4378543, -3.5996532, -2.4417000, 2.4479401
3: -2.4422810, 0.1332530, -2.4340117, 0.1182419, -1.9128861, 1.9201403
4: -7.0440617, -2.8905511, -6.9883175, -2.9186773, -3.2604127, 3.2546067
5: -8.9876633, -5.7355204, -8.9537373, -5.7457619, -2.5401154, 2.5138340
6: -19.4601688, -15.5480824, -19.4427872, -15.5620022, -3.3167620, 3.3163176
7: 4.2270651, 6.9874487, 4.2643237, 6.9667125, -2.7396474, 2.7231250
8: -7.1751165, -4.3977704, -7.1617846, -4.4029832, -2.4648247, 2.4564116
9: -7.2168632, -3.7630327, -7.2016182, -3.7783484, -2.7788677, 2.7780485

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003822, upper bound: 1.3100382
time: 5.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032383, upper bound: 1.3100380
time: 5.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.6044312, -13.5805111, -2.6909666, 2.6909666
1: -10.2822809, -7.4614153, -10.2822809, -7.4614153, -2.3339558, 2.3339558
2: -6.4601760, -3.5581908, -6.4601760, -3.5581908, -2.4652300, 2.4652305
3: -2.4422810, 0.1332530, -2.4422810, 0.1332530, -1.9335856, 1.9335854
4: -7.0440617, -2.8905511, -7.0440617, -2.8905511, -3.2674141, 3.2674150
5: -8.9876633, -5.7355204, -8.9876633, -5.7355204, -2.5519609, 2.5519600
6: -19.4601688, -15.5480824, -19.4601688, -15.5480824, -3.3431511, 3.3431511
7: 4.2270651, 6.9874487, 4.2270651, 6.9874487, -2.7603836, 2.7603836
8: -7.1751165, -4.3977704, -7.1751165, -4.3977704, -2.4760218, 2.4760218
9: -7.2168632, -3.7630327, -7.2168632, -3.7630327, -2.8018560, 2.8018560

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003799, upper bound: 1.3105069
time: 5.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032383, upper bound: 1.3100376
time: 5.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3003824, upper bound: 1.3032384
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3032385, upper bound: 1.3032382
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3003824, upper bound: 1.3032382
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3032385, upper bound: 1.3032381
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3003822, upper bound: 1.3100382
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3032383, upper bound: 1.3100380
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3003799, upper bound: 1.3105069
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.77
Output dim: 7, lower bound: -1.3032383, upper bound: 1.3100376

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.5838776, -13.5928688, -2.6482115, 2.6545534
1: -10.2377262, -7.5479937, -10.2605705, -7.5078421, -2.2533164, 2.2359562
2: -6.3515334, -3.6288996, -6.3999004, -3.6013699, -2.3311529, 2.3443117
3: -2.4029136, 0.1065290, -2.4206142, 0.1164905, -1.8722134, 1.8793442
4: -6.9680395, -2.9741981, -6.9861903, -2.9426339, -3.1814547, 3.1680489
5: -8.9413986, -5.7605448, -8.9513941, -5.7520905, -2.4814978, 2.4850855
6: -19.4147167, -15.5752773, -19.4309235, -15.5635033, -3.2705612, 3.2635984
7: 4.2780347, 6.9527788, 4.2699337, 6.9637675, -2.6857328, 2.6828451
8: -7.1350555, -4.4596825, -7.1578469, -4.4280062, -2.3880091, 2.3897150
9: -7.1800356, -3.7988298, -7.1975260, -3.7873292, -2.7199059, 2.7368531

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919883, upper bound: 1.3001999
time: 5.12 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003779, upper bound: 1.3032366
time: 5.08 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.5882187, -13.5900850, -2.6626835, 2.6638589
1: -10.2623730, -7.4767752, -10.2623739, -7.4767733, -2.3087978, 2.2607450
2: -6.4378519, -3.5996540, -6.4378533, -3.5996532, -2.3407679, 2.4145188
3: -2.4340096, 0.1182394, -2.4340105, 0.1182411, -1.8873348, 1.9047010
4: -6.9883184, -2.9186807, -6.9883199, -2.9186759, -3.2258701, 3.1868277
5: -8.9537363, -5.7457647, -8.9537363, -5.7457628, -2.5024114, 2.4907007
6: -19.4427834, -15.5620041, -19.4427872, -15.5620022, -3.2893085, 3.2962546
7: 4.2643232, 6.9667115, 4.2643242, 6.9667125, -2.7023892, 2.7023873
8: -7.1617851, -4.4029851, -7.1617842, -4.4029841, -2.4472032, 2.4051831
9: -7.2016177, -3.7783499, -7.2016163, -3.7783475, -2.7567048, 2.7679911

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032411, upper bound: 1.3003809
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032389, upper bound: 1.3032385
time: 5.22 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.6000900, -13.5832958, -2.6577969, 2.6713829
1: -10.2377262, -7.5479937, -10.2804909, -7.4924726, -2.2687140, 2.2566867
2: -6.3515334, -3.6288996, -6.4222226, -3.5598989, -2.3596768, 2.3611965
3: -2.4029136, 0.1065290, -2.4289017, 0.1314937, -1.8876438, 1.8875413
4: -6.9680395, -2.9741981, -7.0419283, -2.9144931, -3.2061205, 3.2026038
5: -8.9413986, -5.7605448, -8.9853172, -5.7418528, -2.4929190, 2.5227890
6: -19.4147167, -15.5752773, -19.4483147, -15.5495892, -3.2875443, 3.2810235
7: 4.2780347, 6.9527788, 4.2326822, 6.9844866, -2.7064519, 2.7200966
8: -7.1350555, -4.4596825, -7.1711559, -4.4227934, -2.3936739, 2.4037976
9: -7.1800356, -3.7988298, -7.2127447, -3.7720127, -2.7354441, 2.7531734

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2987938, upper bound: 1.3001998
time: 5.19 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3071780, upper bound: 1.3032362
time: 5.07 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.6044273, -13.5805063, -2.6722713, 2.6806839
1: -10.2623730, -7.4767752, -10.2822809, -7.4614153, -2.3242030, 2.2814636
2: -6.4378519, -3.5996540, -6.4601746, -3.5581923, -2.3684201, 2.4313912
3: -2.4340096, 0.1182394, -2.4422784, 0.1332529, -1.9027729, 1.9128852
4: -6.9883184, -2.9186807, -7.0440621, -2.8905523, -3.2510815, 3.2201943
5: -8.9537363, -5.7457647, -8.9876614, -5.7355204, -2.5138316, 2.5284033
6: -19.4427834, -15.5620041, -19.4601650, -15.5480824, -3.3062687, 3.3136864
7: 4.2643232, 6.9667115, 4.2270665, 6.9874487, -2.7231255, 2.7396450
8: -7.1617851, -4.4029851, -7.1751175, -4.3977699, -2.4528775, 2.4192488
9: -7.2016177, -3.7783499, -7.2168646, -3.7630310, -2.7722397, 2.7842493

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100383, upper bound: 1.3003792
time: 4.85 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100404, upper bound: 1.3032383
time: 5.22 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.5838776, -13.5928688, -2.6650400, 2.6641364
1: -10.2576618, -7.5326185, -10.2605705, -7.5078421, -2.2654066, 2.2513537
2: -6.3738647, -3.5874131, -6.3999004, -3.6013699, -2.3534966, 2.3532701
3: -2.4112453, 0.1215175, -2.4206142, 0.1164905, -1.8804336, 1.8947599
4: -7.0237694, -2.9460731, -6.9861903, -2.9426339, -3.2003193, 3.1968002
5: -8.9753370, -5.7503128, -8.9513941, -5.7520905, -2.5192137, 2.4965014
6: -19.4321251, -15.5613813, -19.4309235, -15.5635033, -3.2880383, 3.2805729
7: 4.2407608, 6.9734745, 4.2699337, 6.9637675, -2.7230067, 2.7035408
8: -7.1483364, -4.4544744, -7.1578469, -4.4280062, -2.4020753, 2.3953743
9: -7.1952000, -3.7835121, -7.1975260, -3.7873292, -2.7361712, 2.7523885

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919898, upper bound: 1.3069983
time: 5.14 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003774, upper bound: 1.3100360
time: 5.21 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5882187, -13.5900850, -2.6795082, 2.6734481
1: -10.2822819, -7.4614186, -10.2623739, -7.4767733, -2.3216000, 2.2761497
2: -6.4601712, -3.5581923, -6.4378533, -3.5996532, -2.3631043, 2.4234846
3: -2.4422777, 0.1332527, -2.4340105, 0.1182411, -1.8955183, 1.9201396
4: -7.0440588, -2.8905525, -6.9883199, -2.9186759, -3.2452726, 3.2155628
5: -8.9876623, -5.7355213, -8.9537363, -5.7457628, -2.5401139, 2.5021195
6: -19.4601593, -15.5480824, -19.4427872, -15.5620022, -3.3067446, 3.3132505
7: 4.2270684, 6.9874468, 4.2643242, 6.9667125, -2.7396441, 2.7231226
8: -7.1751165, -4.3977718, -7.1617842, -4.4029841, -2.4612932, 2.4108639
9: -7.2168646, -3.7630339, -7.2016163, -3.7783475, -2.7730827, 2.7835274

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032406, upper bound: 1.3071805
time: 5.04 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032406, upper bound: 1.3100382
time: 5.23 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.6000900, -13.5832958, -2.6753168, 2.6816568
1: -10.2576618, -7.5326185, -10.2804909, -7.4924726, -2.2784901, 2.2611175
2: -6.3738647, -3.5874131, -6.4222226, -3.5598989, -2.3770323, 2.3732598
3: -2.4112453, 0.1215175, -2.4289017, 0.1314937, -1.9011245, 1.9082184
4: -7.0237694, -2.9460731, -7.0419283, -2.9144931, -3.2230291, 3.2096319
5: -8.9753370, -5.7503128, -8.9853172, -5.7418528, -2.5310583, 2.5346289
6: -19.4321251, -15.5613813, -19.4483147, -15.5495892, -3.3144264, 3.3073983
7: 4.2407608, 6.9734745, 4.2326822, 6.9844866, -2.7437258, 2.7407923
8: -7.1483364, -4.4544744, -7.1711559, -4.4227934, -2.4133573, 2.4150319
9: -7.1952000, -3.7835121, -7.2127447, -3.7720127, -2.7591610, 2.7761621

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2931888, upper bound: 1.3074632
time: 5.17 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3015717, upper bound: 1.3105047
time: 6.26 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.6044273, -13.5805063, -2.6897879, 2.6909647
1: -10.2822819, -7.4614186, -10.2822809, -7.4614153, -2.3339548, 2.2859011
2: -6.4601712, -3.5581923, -6.4601746, -3.5581923, -2.3866358, 2.4434640
3: -2.4422777, 0.1332527, -2.4422784, 0.1332529, -1.9162173, 1.9335849
4: -7.0440588, -2.8905525, -7.0440621, -2.8905523, -3.2674131, 3.2283678
5: -8.9876623, -5.7355213, -8.9876614, -5.7355204, -2.5519581, 2.5402470
6: -19.4601593, -15.5480824, -19.4601650, -15.5480824, -3.3331099, 3.3400869
7: 4.2270684, 6.9874468, 4.2270665, 6.9874487, -2.7603803, 2.7603803
8: -7.1751165, -4.3977718, -7.1751175, -4.3977699, -2.4724927, 2.4304543
9: -7.2168646, -3.7630339, -7.2168646, -3.7630310, -2.7960711, 2.8072400

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3044335, upper bound: 1.3076548
time: 5.51 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3044335, upper bound: 1.3105067
time: 5.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.73 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.2919883, upper bound: 1.3001999
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3003779, upper bound: 1.3032366
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3032411, upper bound: 1.3003809
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3032389, upper bound: 1.3032385
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.2987938, upper bound: 1.3001998
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3071780, upper bound: 1.3032362
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3100383, upper bound: 1.3003792
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3100404, upper bound: 1.3032383
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.2919898, upper bound: 1.3069983
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3003774, upper bound: 1.3100360
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3032406, upper bound: 1.3071805
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3032406, upper bound: 1.3100382
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.2931888, upper bound: 1.3074632
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3015717, upper bound: 1.3105047
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3044335, upper bound: 1.3076548
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 7, lower bound: -1.3044335, upper bound: 1.3105067

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.5736313, -13.6012821, -17.5744324, -13.6146469, -2.6235428, 2.6351378
1: -10.2346964, -7.5493422, -10.2497911, -7.5132103, -2.2444735, 2.2232900
2: -6.3432236, -3.6299334, -6.3711805, -3.6086726, -2.3146915, 2.3125641
3: -2.3879452, 0.1051944, -2.3692198, 0.1059210, -1.8360188, 1.8264995
4: -6.9667368, -2.9809008, -6.9794602, -2.9658582, -3.1571808, 3.1553764
5: -8.9258242, -5.7611694, -8.8975983, -5.7594891, -2.4568672, 2.4315619
6: -19.4131508, -15.5886002, -19.4215336, -15.6093330, -3.2235136, 3.2417111
7: 4.2796860, 6.9420681, 4.2786870, 6.9269180, -2.6472321, 2.6633811
8: -7.1325951, -4.4680557, -7.1457434, -4.4567971, -2.3564105, 2.3672998
9: -7.1783996, -3.8029456, -7.1906233, -3.8017044, -2.7034698, 2.7246156

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2893594, upper bound: 1.2943797
time: 5.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919869, upper bound: 1.3001979
time: 5.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.5838757, -13.5928745, -2.6394377, 2.6537702
1: -10.2377262, -7.5479937, -10.2605686, -7.5078440, -2.2560005, 2.2359548
2: -6.3515334, -3.6288996, -6.3998981, -3.6013691, -2.3311520, 2.3309176
3: -2.4029136, 0.1065290, -2.4206114, 0.1164908, -1.8722134, 1.8388717
4: -6.9680395, -2.9741981, -6.9861894, -2.9426386, -3.1741638, 3.1680489
5: -8.9413986, -5.7605448, -8.9513893, -5.7520933, -2.4814978, 2.4431353
6: -19.4147167, -15.5752773, -19.4309235, -15.5635052, -3.2499781, 3.2635975
7: 4.2780347, 6.9527788, 4.2699342, 6.9637651, -2.6857305, 2.6828446
8: -7.1350555, -4.4596825, -7.1578465, -4.4280081, -2.3681364, 2.3893611
9: -7.1800356, -3.7988298, -7.1975241, -3.7873318, -2.7148738, 2.7368512

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2973497, upper bound: 1.2948414
time: 5.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2973497, upper bound: 1.2948408
time: 5.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900860, -17.5757713, -13.5949659, -2.6591349, 2.6512406
1: -10.2623739, -7.4767752, -10.2377262, -7.5479937, -2.2376504, 2.2607560
2: -6.4378505, -3.5996537, -6.3515334, -3.6288996, -2.3467174, 2.3278341
3: -2.4340081, 0.1182394, -2.4029136, 0.1065290, -1.8875973, 1.8741670
4: -6.9883184, -2.9186816, -6.9680395, -2.9741981, -3.1701212, 3.1879883
5: -8.9537373, -5.7457647, -8.9413986, -5.7605448, -2.4873872, 2.4892550
6: -19.4427814, -15.5620041, -19.4147167, -15.5752773, -3.2823505, 3.2694416
7: 4.2643242, 6.9667120, 4.2780347, 6.9527788, -2.6884546, 2.6886773
8: -7.1617813, -4.4029856, -7.1350555, -4.4596825, -2.3908195, 2.4000399
9: -7.2016172, -3.7783518, -7.1800356, -3.7988298, -2.7367439, 2.7341056

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3002024, upper bound: 1.2919880
time: 5.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032367, upper bound: 1.3003794
time: 5.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.5882168, -13.5900850, -2.6626811, 2.6626806
1: -10.2623730, -7.4767752, -10.2623730, -7.4767752, -2.2607450, 2.2607450
2: -6.4378519, -3.5996540, -6.4378519, -3.5996540, -2.3407669, 2.3407669
3: -2.4340096, 0.1182394, -2.4340096, 0.1182394, -1.8873343, 1.8873339
4: -6.9883184, -2.9186807, -6.9883184, -2.9186807, -3.1868267, 3.1868262
5: -8.9537363, -5.7457647, -8.9537363, -5.7457647, -2.4907007, 2.4907012
6: -19.4427834, -15.5620041, -19.4427834, -15.5620041, -3.2893066, 3.2893071
7: 4.2643232, 6.9667115, 4.2643232, 6.9667115, -2.7023883, 2.7023883
8: -7.1617851, -4.4029851, -7.1617851, -4.4029851, -2.4051814, 2.4051809
9: -7.2016177, -3.7783499, -7.2016177, -3.7783499, -2.7679873, 2.7679877

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3002005, upper bound: 1.2919877
time: 5.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032370, upper bound: 1.3003794
time: 5.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.5736313, -13.6012821, -17.5905819, -13.6050854, -2.6331139, 2.6505561
1: -10.2346964, -7.5493422, -10.2697411, -7.4978881, -2.2591457, 2.2440648
2: -6.3432236, -3.6299334, -6.3934956, -3.5671959, -2.3301282, 2.3294487
3: -2.3879452, 0.1051944, -2.3775716, 0.1209025, -1.8499045, 1.8347821
4: -6.9667368, -2.9809008, -7.0352135, -2.9377854, -3.1817231, 3.1802611
5: -8.9258242, -5.7611694, -8.9316196, -5.7492590, -2.4652948, 2.4693613
6: -19.4131508, -15.5886002, -19.4389534, -15.5954704, -3.2404413, 3.2592206
7: 4.2796860, 6.9420681, 4.2413874, 6.9475851, -2.6678991, 2.7006807
8: -7.1325951, -4.4680557, -7.1590023, -4.4515939, -2.3620658, 2.3813307
9: -7.1783996, -3.8029456, -7.2057853, -3.7863569, -2.7190375, 2.7408705

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2961438, upper bound: 1.2943789
time: 4.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2987924, upper bound: 1.3001983
time: 5.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949659, -17.6000862, -13.5832968, -2.6490197, 2.6705992
1: -10.2377262, -7.5479937, -10.2804880, -7.4924722, -2.2713962, 2.2566862
2: -6.3515334, -3.6288996, -6.4222217, -3.5598974, -2.3566904, 2.3478019
3: -2.4029136, 0.1065290, -2.4288974, 0.1314951, -1.8876443, 1.8470683
4: -6.9680395, -2.9741981, -7.0419269, -2.9144959, -3.1988230, 3.2009215
5: -8.9413986, -5.7605448, -8.9853115, -5.7418528, -2.4929190, 2.4808373
6: -19.4147167, -15.5752773, -19.4483109, -15.5495920, -3.2669611, 3.2810225
7: 4.2780347, 6.9527788, 4.2326822, 6.9844851, -2.7064505, 2.7200966
8: -7.1350555, -4.4596825, -7.1711545, -4.4227972, -2.3738041, 2.4034455
9: -7.1800356, -3.7988298, -7.2127447, -3.7720141, -2.7304120, 2.7531738

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3041484, upper bound: 1.2948404
time: 5.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3041484, upper bound: 1.2948402
time: 6.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900860, -17.5919857, -13.5853930, -2.6687174, 2.6680694
1: -10.2623739, -7.4767752, -10.2576618, -7.5326185, -2.2530484, 2.2692037
2: -6.4378505, -3.5996537, -6.3738647, -3.5874131, -2.3556752, 2.3447154
3: -2.4340081, 0.1182394, -2.4112453, 0.1215175, -1.9014840, 1.8823869
4: -6.9883184, -2.9186816, -7.0237694, -2.9460731, -3.1953239, 3.2035084
5: -8.9537373, -5.7457647, -8.9753370, -5.7503128, -2.4988041, 2.5254054
6: -19.4427814, -15.5620041, -19.4321251, -15.5613813, -3.2993250, 3.2869196
7: 4.2643242, 6.9667120, 4.2407608, 6.9734745, -2.7091503, 2.7259512
8: -7.1617813, -4.4029856, -7.1483364, -4.4544744, -2.3964787, 2.4128809
9: -7.2016172, -3.7783518, -7.1952000, -3.7835121, -2.7522798, 2.7503700

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3069979, upper bound: 1.2919869
time: 5.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100361, upper bound: 1.3003770
time: 5.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.5882168, -13.5900850, -17.6044273, -13.5805111, -2.6722708, 2.6795058
1: -10.2623730, -7.4767752, -10.2822819, -7.4614186, -2.2761497, 2.2814636
2: -6.4378519, -3.5996540, -6.4601712, -3.5581923, -2.3656988, 2.3631039
3: -2.4340096, 0.1182394, -2.4422777, 0.1332527, -1.9027719, 1.8955178
4: -6.9883184, -2.9186807, -7.0440588, -2.8905525, -3.2155609, 3.2165790
5: -8.9537363, -5.7457647, -8.9876623, -5.7355213, -2.5021195, 2.5284033
6: -19.4427834, -15.5620041, -19.4601593, -15.5480824, -3.3062668, 3.3067427
7: 4.2643232, 6.9667115, 4.2270684, 6.9874468, -2.7231236, 2.7396431
8: -7.1617851, -4.4029851, -7.1751165, -4.3977718, -2.4108615, 2.4192464
9: -7.2016177, -3.7783499, -7.2168646, -3.7630339, -2.7835236, 2.7842469

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3069982, upper bound: 1.2919869
time: 5.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100363, upper bound: 1.3003770
time: 5.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.5898266, -13.5917072, -17.5744324, -13.6146469, -2.6403542, 2.6443167
1: -10.2546425, -7.5339794, -10.2497911, -7.5132103, -2.2548549, 2.2386742
2: -6.3655529, -3.5884447, -6.3711805, -3.6086726, -2.3370342, 2.3215179
3: -2.3962955, 0.1201743, -2.3692198, 0.1059210, -1.8426011, 1.8419111
4: -7.0224776, -2.9527938, -6.9794602, -2.9658582, -3.1759663, 3.1841059
5: -8.9597912, -5.7509413, -8.8975983, -5.7594891, -2.4775410, 2.4429750
6: -19.4305687, -15.5747156, -19.4215336, -15.6093330, -3.2410297, 3.2586675
7: 4.2423983, 6.9627457, 4.2786870, 6.9269180, -2.6845198, 2.6840587
8: -7.1458631, -4.4628515, -7.1457434, -4.4567971, -2.3704662, 2.3729558
9: -7.1935434, -3.7876177, -7.1906233, -3.8017044, -2.7197151, 2.7401605

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2893590, upper bound: 1.3011580
time: 5.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919865, upper bound: 1.3069963
time: 5.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.5838757, -13.5928745, -2.6562614, 2.6633532
1: -10.2576618, -7.5326185, -10.2605686, -7.5078440, -2.2674787, 2.2513528
2: -6.3738647, -3.5874131, -6.3998981, -3.6013691, -2.3534956, 2.3398757
3: -2.4112453, 0.1215175, -2.4206114, 0.1164908, -1.8804331, 1.8542881
4: -7.0237694, -2.9460731, -6.9861894, -2.9426386, -3.1930208, 3.1968007
5: -8.9753370, -5.7503128, -8.9513893, -5.7520933, -2.5191345, 2.4545512
6: -19.4321251, -15.5613813, -19.4309235, -15.5635052, -3.2674570, 3.2805719
7: 4.2407608, 6.9734745, 4.2699342, 6.9637651, -2.7230043, 2.7035403
8: -7.1483364, -4.4544744, -7.1578465, -4.4280081, -2.3821874, 2.3950205
9: -7.1952000, -3.7835121, -7.1975241, -3.7873318, -2.7311387, 2.7523866

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2973493, upper bound: 1.3016434
time: 5.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2973493, upper bound: 1.3100366
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5757713, -13.5949659, -2.6759596, 2.6608298
1: -10.2822781, -7.4614172, -10.2377262, -7.5479937, -2.2504079, 2.2735639
2: -6.4601707, -3.5581920, -6.3515334, -3.6288996, -2.3636127, 2.3367994
3: -2.4422765, 0.1332510, -2.4029136, 0.1065290, -1.8942277, 1.8896055
4: -7.0440602, -2.8905554, -6.9680395, -2.9741981, -3.1895213, 3.2093201
5: -8.9876614, -5.7355223, -8.9413986, -5.7605448, -2.5250888, 2.5006771
6: -19.4601612, -15.5480824, -19.4147167, -15.5752773, -3.2997837, 3.2864375
7: 4.2270703, 6.9874458, 4.2780347, 6.9527788, -2.7257085, 2.7094111
8: -7.1751146, -4.3977728, -7.1350555, -4.4596825, -2.4049101, 2.4055953
9: -7.2168636, -3.7630343, -7.1800356, -3.7988298, -2.7531214, 2.7496409

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3002019, upper bound: 1.2987937
time: 5.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032362, upper bound: 1.3071793
time: 5.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5882168, -13.5900850, -2.6795058, 2.6722705
1: -10.2822819, -7.4614186, -10.2623730, -7.4767752, -2.2814636, 2.2761493
2: -6.4601712, -3.5581923, -6.4378519, -3.5996540, -2.3631039, 2.3656988
3: -2.4422777, 0.1332527, -2.4340096, 0.1182394, -1.8955178, 1.9027719
4: -7.0440588, -2.8905525, -6.9883184, -2.9186807, -3.2165790, 3.2155614
5: -8.9876623, -5.7355213, -8.9537363, -5.7457647, -2.5284033, 2.5021195
6: -19.4601593, -15.5480824, -19.4427834, -15.5620041, -3.3067427, 3.3062663
7: 4.2270684, 6.9874468, 4.2643232, 6.9667115, -2.7396431, 2.7231236
8: -7.1751165, -4.3977718, -7.1617851, -4.4029851, -2.4192467, 2.4108617
9: -7.2168646, -3.7630339, -7.2016177, -3.7783499, -2.7842474, 2.7835236

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3002000, upper bound: 1.2987935
time: 7.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032365, upper bound: 1.3071793
time: 5.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.5898266, -13.5917072, -17.5905819, -13.6050854, -2.6506176, 2.6622369
1: -10.2546425, -7.5339794, -10.2697411, -7.4978881, -2.2688732, 2.2484818
2: -6.3655529, -3.5884447, -6.3934956, -3.5671959, -2.3501205, 2.3415046
3: -2.3962955, 0.1201743, -2.3775716, 0.1209025, -1.8600440, 1.8554540
4: -7.0224776, -2.9527938, -7.0352135, -2.9377854, -3.1986117, 3.1969562
5: -8.9597912, -5.7509413, -8.9316196, -5.7492590, -2.4891310, 2.4811993
6: -19.4305687, -15.5747156, -19.4389534, -15.5954704, -3.2673607, 3.2852936
7: 4.2423983, 6.9627457, 4.2413874, 6.9475851, -2.7051868, 2.7213583
8: -7.1458631, -4.4628515, -7.1590023, -4.4515939, -2.3817668, 2.3926587
9: -7.1935434, -3.7876177, -7.2057853, -3.7863569, -2.7427378, 2.7638698

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2905237, upper bound: 1.3015989
time: 5.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919865, upper bound: 1.3069959
time: 9.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5919857, -13.5853930, -17.6000862, -13.5832968, -2.6665339, 2.6808734
1: -10.2576618, -7.5326185, -10.2804880, -7.4924722, -2.2811718, 2.2611179
2: -6.3738647, -3.5874131, -6.4222217, -3.5598974, -2.3766775, 2.3598659
3: -2.4112453, 0.1215175, -2.4288974, 0.1314951, -1.9011245, 1.8677449
4: -7.0237694, -2.9460731, -7.0419269, -2.9144959, -3.2157316, 3.2096319
5: -8.9753370, -5.7503128, -8.9853115, -5.7418528, -2.5307302, 2.4926772
6: -19.4321251, -15.5613813, -19.4483109, -15.5495920, -3.2938423, 3.3073988
7: 4.2407608, 6.9734745, 4.2326822, 6.9844851, -2.7437243, 2.7407923
8: -7.1483364, -4.4544744, -7.1711545, -4.4227972, -2.3934722, 2.4146805
9: -7.1952000, -3.7835121, -7.2127447, -3.7720141, -2.7541308, 2.7761621

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2985427, upper bound: 1.3021111
time: 6.13 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2985427, upper bound: 1.3105048
time: 5.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5919857, -13.5853930, -2.6862335, 2.6783504
1: -10.2822781, -7.4614172, -10.2576618, -7.5326185, -2.2627997, 2.2833061
2: -6.4601707, -3.5581920, -6.3738647, -3.5874131, -2.3756781, 2.3567863
3: -2.4422765, 0.1332510, -2.4112453, 0.1215175, -1.9116721, 1.9030859
4: -7.0440602, -2.8905554, -7.0237694, -2.9460731, -3.2116804, 3.2262306
5: -8.9876614, -5.7355223, -8.9753370, -5.7503128, -2.5369291, 2.5370021
6: -19.4601612, -15.5480824, -19.4321251, -15.5613813, -3.3261604, 3.3133183
7: 4.2270703, 6.9874458, 4.2407608, 6.9734745, -2.7464042, 2.7466850
8: -7.1751146, -4.3977728, -7.1483364, -4.4544744, -2.4160962, 2.4228797
9: -7.2168636, -3.7630343, -7.1952000, -3.7835121, -2.7761087, 2.7733588

Time for backsubstitution: 14.41 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7229785919189453
rel_dist={7: [-1.3105157996816672, 1.3105151877312444]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 457
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1841506, upper bound: 1.1791578
time: 7.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847103, upper bound: 1.1847087
time: 4.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.67
Output dim: 7, lower bound: -1.1841506, upper bound: 1.1791578
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.67
Output dim: 7, lower bound: -1.1847103, upper bound: 1.1847087

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.5930252, -13.5878286, -2.5169859, 2.5178189
1: -10.2623758, -7.4767718, -10.2640038, -7.4714136, -2.2490292, 2.2455969
2: -6.4378543, -3.5996532, -6.4474549, -3.5983844, -2.3524599, 2.3608418
3: -2.4340117, 0.1182419, -2.4360168, 0.1221890, -1.8353753, 1.8326535
4: -6.9883175, -2.9186773, -6.9913082, -2.9069729, -3.1420984, 3.1336884
5: -8.9537373, -5.7457619, -8.9571953, -5.7410479, -2.4210677, 2.4190974
6: -19.4427872, -15.5620022, -19.4446373, -15.5569849, -3.1816692, 3.1770754
7: 4.2643237, 6.9667125, 4.2619162, 6.9752636, -2.7109399, 2.7047963
8: -7.1617846, -4.4029832, -7.1654892, -4.4018087, -2.3793969, 2.3808315
9: -7.2016182, -3.7783484, -7.2060962, -3.7777143, -2.6713319, 2.6756616

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791545
time: 4.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791550
time: 5.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -17.6044312, -13.5805111, -17.5972481, -13.5857983, -2.5366158, 2.5315759
1: -10.2822809, -7.4614153, -10.2654266, -7.4666910, -2.2746015, 2.2608061
2: -6.4601760, -3.5581908, -6.4559088, -3.5972753, -2.3724957, 2.3958044
3: -2.4422810, 0.1332530, -2.4377637, 0.1256830, -1.8518310, 1.8501282
4: -7.0440617, -2.8905511, -6.9938722, -2.8966670, -3.1867442, 3.1572790
5: -8.9876633, -5.7355204, -8.9602032, -5.7368994, -2.4638071, 2.4334917
6: -19.4601688, -15.5480824, -19.4462547, -15.5525627, -3.2123318, 3.1964064
7: 4.2270651, 6.9874487, 4.2598319, 6.9827957, -2.7557306, 2.7276168
8: -7.1751165, -4.3977704, -7.1687713, -4.4007754, -2.3974924, 2.3905525
9: -7.2168632, -3.7630327, -7.2100444, -3.7771642, -2.6893601, 2.7021303

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847092, upper bound: 1.1825615
time: 5.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847092, upper bound: 1.1847075
time: 5.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.22 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 25.22
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791545
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 25.22
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791550
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.22
Output dim: 7, lower bound: -1.1847092, upper bound: 1.1825615
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.22
Output dim: 7, lower bound: -1.1847092, upper bound: 1.1847075

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -17.5992393, -13.5838442, -17.5848007, -13.5906773, -2.5264196, 2.5153303
1: -10.2801323, -7.4985771, -10.2407970, -7.5379014, -2.2014408, 2.1992197
2: -6.4147625, -3.5602381, -6.3695850, -3.6265092, -2.2906842, 2.3072228
3: -2.4262762, 0.1311442, -2.4066968, 0.1139741, -1.8238645, 1.8172219
4: -7.0415072, -2.9192050, -6.9735947, -2.9521720, -3.1285372, 3.1081491
5: -8.9848461, -5.7430944, -8.9478683, -5.7516861, -2.4460173, 2.4110451
6: -19.4459839, -15.5498953, -19.4181862, -15.5658512, -3.1758585, 3.1672335
7: 4.2337818, 6.9838934, 4.2735133, 6.9688673, -2.7350855, 2.7103801
8: -7.1703596, -4.4277186, -7.1420612, -4.4574776, -2.3355894, 2.3215568
9: -7.2119207, -3.7737772, -7.1884127, -3.7976506, -2.6625123, 2.6593046

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809639, upper bound: 1.1748100
time: 6.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847070, upper bound: 1.1825588
time: 5.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044235, -13.5805092, -17.5972462, -13.5857992, -2.5366154, 2.5303400
1: -10.2822809, -7.4614162, -10.2654285, -7.4666920, -2.2241421, 2.2608042
2: -6.4601746, -3.5581908, -6.4559054, -3.5972748, -2.3656554, 2.3116658
3: -2.4422793, 0.1332521, -2.4377630, 0.1256825, -1.8518295, 1.8318930
4: -7.0440602, -2.8905487, -6.9938698, -2.8966689, -3.1436701, 3.1572776
5: -8.9876633, -5.7355213, -8.9602032, -5.7369003, -2.4513283, 2.4334898
6: -19.4601669, -15.5480824, -19.4462452, -15.5525627, -3.2092590, 3.1842594
7: 4.2270679, 6.9874463, 4.2598333, 6.9827919, -2.7557240, 2.7276130
8: -7.1751161, -4.3977699, -7.1687722, -4.4007788, -2.3481827, 2.3870080
9: -7.2168646, -3.7630310, -7.2100434, -3.7771673, -2.6919289, 2.6963215

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809639, upper bound: 1.1769543
time: 8.15 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847070, upper bound: 1.1847047
time: 5.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.21 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 28.21
Output dim: 7, lower bound: -1.1809639, upper bound: 1.1748100
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -1.1847070, upper bound: 1.1825588
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 28.21
Output dim: 7, lower bound: -1.1809639, upper bound: 1.1769543
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -1.1847070, upper bound: 1.1847047

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.5992374, -13.5838499, -17.5848007, -13.5906773, -2.5256367, 2.5055854
1: -10.2801323, -7.4985790, -10.2407970, -7.5379014, -2.2014399, 2.2017984
2: -6.4147611, -3.5602391, -6.3695850, -3.6265092, -2.2766194, 2.3042369
3: -2.4262710, 0.1311431, -2.4066968, 0.1139741, -1.7813668, 1.8172214
4: -7.0415025, -2.9192057, -6.9735947, -2.9521720, -3.1268549, 3.1004963
5: -8.9848442, -5.7430954, -8.9478683, -5.7516861, -2.4019704, 2.4110451
6: -19.4459820, -15.5498962, -19.4181862, -15.5658512, -3.1758585, 3.1456213
7: 4.2337818, 6.9838905, 4.2735133, 6.9688673, -2.7350855, 2.7103772
8: -7.1703577, -4.4277215, -7.1420612, -4.4574776, -2.3352380, 2.3004231
9: -7.2119188, -3.7737784, -7.1884127, -3.7976506, -2.6625109, 2.6540198

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1819985
time: 5.61 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1825597
time: 4.97 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.5972462, -13.5857992, -2.5358315, 2.5205889
1: -10.2822781, -7.4614172, -10.2654285, -7.4666920, -2.2241402, 2.2633829
2: -6.4601707, -3.5581915, -6.4559054, -3.5972748, -2.3515892, 2.3086774
3: -2.4422753, 0.1332524, -2.4377630, 0.1256825, -1.8093333, 1.8318913
4: -7.0440583, -2.8905542, -6.9938698, -2.8966689, -3.1419878, 3.1496248
5: -8.9876595, -5.7355223, -8.9602032, -5.7369003, -2.4072804, 2.4334888
6: -19.4601650, -15.5480843, -19.4462452, -15.5525627, -3.2092590, 3.1626492
7: 4.2270660, 6.9874463, 4.2598333, 6.9827919, -2.7557259, 2.7276130
8: -7.1751146, -4.3977728, -7.1687722, -4.4007788, -2.3478303, 2.3658903
9: -7.2168641, -3.7630334, -7.2100434, -3.7771673, -2.6919279, 2.6910357

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1841449
time: 5.41 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1847055
time: 5.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.84 seconds
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.84
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1819985
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.84
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1825597
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.84
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1841449
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.84
Output dim: 7, lower bound: -1.1791523, upper bound: 1.1847055

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.5992374, -13.5838499, -17.5919857, -13.5853930, -2.5290427, 2.5139790
1: -10.2801323, -7.4985790, -10.2576618, -7.5326185, -2.1949568, 2.2091341
2: -6.4147611, -3.5602391, -6.3738647, -3.5874131, -2.2830801, 2.3050323
3: -2.4262710, 0.1311431, -2.4112453, 0.1215175, -1.7885547, 1.8261905
4: -7.0415025, -2.9192057, -7.0237694, -2.9460731, -3.1112127, 3.1126699
5: -8.9848442, -5.7430954, -8.9753370, -5.7503128, -2.4033799, 2.4418314
6: -19.4459820, -15.5498962, -19.4321251, -15.5613813, -3.1814079, 3.1671491
7: 4.2337818, 6.9838905, 4.2407608, 6.9734745, -2.7396927, 2.7431297
8: -7.1703577, -4.4277215, -7.1483364, -4.4544744, -2.3386469, 2.3107841
9: -7.2119188, -3.7737784, -7.1952000, -3.7835121, -2.6816397, 2.6602974

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1760458, upper bound: 1.1770330
time: 6.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791511, upper bound: 1.1825582
time: 5.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6042671, -13.5809402, -17.5882168, -13.5900850, -2.5290351, 2.5114901
1: -10.2821770, -7.4614344, -10.2623730, -7.4767752, -2.2136755, 2.2596810
2: -6.4601297, -3.5582700, -6.4378519, -3.5996540, -2.3447819, 2.2903469
3: -2.4422748, 0.1331892, -2.4340096, 0.1182394, -1.7960219, 1.8274796
4: -7.0437407, -2.8905916, -6.9883184, -2.9186807, -3.1192117, 3.1443839
5: -8.9874916, -5.7355232, -8.9537363, -5.7457647, -2.3966928, 2.4271030
6: -19.4600716, -15.5481157, -19.4427834, -15.5620041, -3.1885328, 3.1575975
7: 4.2272778, 6.9874015, 4.2643232, 6.9667115, -2.7394338, 2.7230783
8: -7.1750178, -4.3978462, -7.1617851, -4.4029851, -2.3405027, 2.3571091
9: -7.2168140, -3.7635975, -7.2016177, -3.7783499, -2.6882715, 2.6730127

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1760480, upper bound: 1.1786111
time: 5.27 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791511, upper bound: 1.1841432
time: 5.08 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805111, -17.6044273, -13.5805111, -2.5392475, 2.5289903
1: -10.2822781, -7.4614172, -10.2822819, -7.4614186, -2.2176704, 2.2699370
2: -6.4601707, -3.5581915, -6.4601712, -3.5581923, -2.3580675, 2.3094623
3: -2.4422753, 0.1332524, -2.4422777, 0.1332527, -1.8165445, 1.8408065
4: -7.0440583, -2.8905542, -7.0440588, -2.8905525, -3.1284046, 3.1617498
5: -8.9876595, -5.7355223, -8.9876623, -5.7355213, -2.4086981, 2.4619017
6: -19.4601650, -15.5480843, -19.4601593, -15.5480824, -3.2148199, 3.1841359
7: 4.2270660, 6.9874463, 4.2270684, 6.9874468, -2.7603807, 2.7603779
8: -7.1751146, -4.3977728, -7.1751165, -4.3977718, -2.3512545, 2.3762202
9: -7.2168641, -3.7630334, -7.2168646, -3.7630339, -2.7110591, 2.6974134

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1760458, upper bound: 1.1705253
time: 7.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791511, upper bound: 1.1847041
time: 5.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 27.54 seconds
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.54
Output dim: 7, lower bound: -1.1760458, upper bound: 1.1770330
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.54
Output dim: 7, lower bound: -1.1791511, upper bound: 1.1825582
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 27.54
Output dim: 7, lower bound: -1.1760480, upper bound: 1.1786111
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.54
Output dim: 7, lower bound: -1.1791511, upper bound: 1.1841432
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.54
Output dim: 7, lower bound: -1.1760458, upper bound: 1.1705253
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.54
Output dim: 7, lower bound: -1.1791511, upper bound: 1.1847041

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.5992355, -13.5838594, -17.5919857, -13.5853930, -2.5271044, 2.4647279
1: -10.2801294, -7.4985862, -10.2576618, -7.5326195, -2.1949553, 2.1827569
2: -6.4147587, -3.5602379, -6.3738637, -3.5874128, -2.2309618, 2.2954609
3: -2.4262674, 0.1311431, -2.4112463, 0.1215165, -1.7590899, 1.8223746
4: -7.0415030, -2.9192059, -7.0237699, -2.9460714, -3.1112108, 3.0955276
5: -8.9848404, -5.7430954, -8.9753361, -5.7503128, -2.3850951, 2.4370279
6: -19.4459801, -15.5499010, -19.4321251, -15.5613804, -3.1814060, 3.1382790
7: 4.2337837, 6.9838877, 4.2407613, 6.9734740, -2.7396903, 2.7431264
8: -7.1703563, -4.4277229, -7.1483383, -4.4544740, -2.3386440, 2.3078418
9: -7.2119193, -3.7737811, -7.1952009, -3.7835126, -2.6813865, 2.6379724

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1788137
time: 5.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1825590
time: 6.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6042633, -13.5809536, -17.5882187, -13.5900860, -2.5290356, 2.4622381
1: -10.2821751, -7.4614406, -10.2623758, -7.4767742, -2.2136750, 2.2331388
2: -6.4601293, -3.5582700, -6.4378519, -3.5996542, -2.2926493, 2.2807670
3: -2.4422698, 0.1331861, -2.4340084, 0.1182408, -1.7664919, 1.8254187
4: -7.0437365, -2.8905921, -6.9883180, -2.9186819, -3.1153712, 3.1269455
5: -8.9874849, -5.7355251, -8.9537392, -5.7457647, -2.3783727, 2.4271026
6: -19.4600677, -15.5481215, -19.4427834, -15.5620022, -3.1885347, 3.1287279
7: 4.2272778, 6.9874015, 4.2643251, 6.9667120, -2.7394342, 2.7230763
8: -7.1750159, -4.3978481, -7.1617827, -4.4029856, -2.3405008, 2.3541598
9: -7.2168150, -3.7636032, -7.2016172, -3.7783487, -2.6880164, 2.6506658

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1713149, upper bound: 1.1803296
time: 5.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1713149, upper bound: 1.1841444
time: 5.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805244, -17.6044254, -13.5805101, -2.5392456, 2.4797380
1: -10.2822790, -7.4614239, -10.2822800, -7.4614172, -2.2176695, 2.2433949
2: -6.4601674, -3.5581915, -6.4601707, -3.5581918, -2.3059344, 2.2998834
3: -2.4422705, 0.1332501, -2.4422789, 0.1332517, -1.7870140, 1.8360784
4: -7.0440559, -2.8905544, -7.0440602, -2.8905551, -3.1284046, 3.1443839
5: -8.9876537, -5.7355242, -8.9876614, -5.7355213, -2.3903770, 2.4570980
6: -19.4601650, -15.5480900, -19.4601612, -15.5480843, -3.2148180, 3.1552677
7: 4.2270679, 6.9874449, 4.2270679, 6.9874477, -2.7603798, 2.7603769
8: -7.1751142, -4.3977742, -7.1751165, -4.3977723, -2.3512530, 2.3732722
9: -7.2168651, -3.7630363, -7.2168641, -3.7630334, -2.7108059, 2.6750660

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1809608
time: 5.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1847047
time: 5.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 26.05 seconds
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 26.05
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1788137
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 26.05
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1825590
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 26.05
Output dim: 7, lower bound: -1.1713149, upper bound: 1.1803296
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 26.05
Output dim: 7, lower bound: -1.1713149, upper bound: 1.1841444
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 26.05
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1809608
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 26.05
Output dim: 7, lower bound: -1.1724788, upper bound: 1.1847047

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.5992355, -13.5838594, -17.5919857, -13.5853920, -2.5191872, 2.4647269
1: -10.2801294, -7.4985862, -10.2576609, -7.5326195, -2.1975350, 2.1827555
2: -6.4147587, -3.5602379, -6.3738618, -3.5874126, -2.2294750, 2.2828927
3: -2.4262674, 0.1311431, -2.4112411, 0.1215169, -1.7588406, 1.7836924
4: -7.0415030, -2.9192059, -7.0237727, -2.9460754, -3.1035576, 3.0954781
5: -8.9848404, -5.7430954, -8.9753323, -5.7503138, -2.3850946, 2.3976278
6: -19.4459801, -15.5499010, -19.4321270, -15.5613861, -3.1597967, 3.1382794
7: 4.2337837, 6.9838877, 4.2407618, 6.9734697, -2.7396860, 2.7431259
8: -7.1703563, -4.4277229, -7.1483359, -4.4544764, -2.3178272, 2.3078392
9: -7.2119193, -3.7737811, -7.1951985, -3.7835131, -2.6761022, 2.6379714

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1705445, upper bound: 1.1825581
time: 4.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1705445, upper bound: 1.1825588
time: 5.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.6042633, -13.5809536, -17.5882149, -13.5900879, -2.5200181, 2.4622369
1: -10.2821751, -7.4614406, -10.2623739, -7.4767756, -2.2158637, 2.2331376
2: -6.4601293, -3.5582700, -6.4378490, -3.5996542, -2.2911611, 2.2681994
3: -2.4422698, 0.1331861, -2.4340043, 0.1182404, -1.7664914, 1.7849820
4: -7.0437365, -2.8905921, -6.9883184, -2.9186826, -3.1085253, 3.1260829
5: -8.9874849, -5.7355251, -8.9537334, -5.7457647, -2.3783731, 2.3830547
6: -19.4600677, -15.5481215, -19.4427814, -15.5620041, -3.1669235, 3.1287279
7: 4.2272778, 6.9874015, 4.2643251, 6.9667082, -2.7394304, 2.7230763
8: -7.1750159, -4.3978481, -7.1617832, -4.4029875, -2.3196921, 2.3541589
9: -7.2168150, -3.7636032, -7.2016153, -3.7783515, -2.6827335, 2.6506648

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1691718, upper bound: 1.1841442
time: 5.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1691698, upper bound: 1.1723939
time: 9.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805244, -17.6044235, -13.5805130, -2.5302229, 2.4797375
1: -10.2822790, -7.4614239, -10.2822790, -7.4614182, -2.2202473, 2.2433944
2: -6.4601674, -3.5581915, -6.4601688, -3.5581913, -2.3044443, 2.2873163
3: -2.4422705, 0.1332501, -2.4422750, 0.1332523, -1.7870140, 1.7980528
4: -7.0440559, -2.8905544, -7.0440583, -2.8905554, -3.1207495, 3.1435184
5: -8.9876537, -5.7355242, -8.9876575, -5.7355251, -2.3903761, 2.4176977
6: -19.4601650, -15.5480900, -19.4601612, -15.5480843, -3.1932087, 3.1552668
7: 4.2270679, 6.9874449, 4.2270689, 6.9874458, -2.7603779, 2.7603760
8: -7.1751142, -4.3977742, -7.1751146, -4.3977733, -2.3304486, 2.3732703
9: -7.2168651, -3.7630363, -7.2168622, -3.7630348, -2.7055206, 2.6750650

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1703331, upper bound: 1.1847046
time: 5.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1703331, upper bound: 1.1825588
time: 5.27 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 25.01 seconds
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.01
Output dim: 7, lower bound: -1.1705445, upper bound: 1.1825581
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.01
Output dim: 7, lower bound: -1.1705445, upper bound: 1.1825588
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.01
Output dim: 7, lower bound: -1.1691718, upper bound: 1.1841442
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 25.01
Output dim: 7, lower bound: -1.1691698, upper bound: 1.1723939
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.01
Output dim: 7, lower bound: -1.1703331, upper bound: 1.1847046
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.01
Output dim: 7, lower bound: -1.1703331, upper bound: 1.1825588

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.5919838, -13.5854034, -17.5919857, -13.5853920, -2.5128961, 2.4636469
1: -10.2576609, -7.5326247, -10.2576609, -7.5326195, -2.1751666, 2.1487894
2: -6.3738594, -3.5874147, -6.3738618, -3.5874126, -2.2131896, 2.2557290
3: -2.4112382, 0.1215169, -2.4112411, 0.1215169, -1.7446237, 1.7740293
4: -7.0237675, -2.9460752, -7.0237727, -2.9460754, -3.0857525, 3.0686121
5: -8.9753265, -5.7503138, -8.9753323, -5.7503138, -2.3746958, 2.3930020
6: -19.4321270, -15.5613918, -19.4321270, -15.5613861, -3.1525202, 3.1236501
7: 4.2407608, 6.9734697, 4.2407618, 6.9734697, -2.7327089, 2.7327080
8: -7.1483359, -4.4544773, -7.1483359, -4.4544764, -2.2925982, 2.2896533
9: -7.1951995, -3.7835174, -7.1951985, -3.7835131, -2.6544957, 2.6324468

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1694050, upper bound: 1.1794633
time: 5.48 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1694049, upper bound: 1.1825593
time: 5.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044235, -13.5805264, -17.5919857, -13.5853920, -2.5215566, 2.4683521
1: -10.2822781, -7.4614239, -10.2576609, -7.5326195, -2.1919551, 2.1877952
2: -6.4601665, -3.5581932, -6.3738618, -3.5874126, -2.2323575, 2.2603192
3: -2.4422700, 0.1332493, -2.4112411, 0.1215169, -1.7623177, 1.7799511
4: -7.0440555, -2.8905609, -7.0237727, -2.9460754, -3.1013517, 3.0992947
5: -8.9876518, -5.7355242, -8.9753323, -5.7503138, -2.3878202, 2.3992200
6: -19.4601593, -15.5480938, -19.4321270, -15.5613861, -3.1748018, 3.1375709
7: 4.2270689, 6.9874415, 4.2407618, 6.9734697, -2.7464008, 2.7466798
8: -7.1751122, -4.3977766, -7.1483359, -4.4544764, -2.3198190, 2.3183403
9: -7.2168627, -3.7630401, -7.1951985, -3.7835131, -2.6771998, 2.6523509

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1694049, upper bound: 1.1794644
time: 5.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1694049, upper bound: 1.1825592
time: 5.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.5918236, -13.5858364, -17.5882111, -13.5900917, -2.5073972, 2.4587648
1: -10.2575569, -7.5326424, -10.2623720, -7.4767737, -2.1939363, 2.1619463
2: -6.3738203, -3.5874915, -6.4378486, -3.5996544, -2.2045116, 2.2557824
3: -2.4112363, 0.1214521, -2.4340031, 0.1182401, -1.7361135, 1.7744508
4: -7.0234480, -2.9461112, -6.9883165, -2.9186835, -3.0921769, 3.0703464
5: -8.9751596, -5.7503138, -8.9537306, -5.7457647, -2.3727841, 2.3680258
6: -19.4320297, -15.5614185, -19.4427795, -15.5620041, -3.1401567, 3.1238794
7: 4.2409711, 6.9734287, 4.2643237, 6.9667082, -2.7257371, 2.7091050
8: -7.1482363, -4.4545512, -7.1617823, -4.4029880, -2.3096151, 2.2977629
9: -7.1951523, -3.7840834, -7.2016168, -3.7783520, -2.6516643, 2.6307554

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1681052, upper bound: 1.1810432
time: 5.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1681049, upper bound: 1.1841437
time: 5.25 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.5919838, -13.5854034, -17.6044235, -13.5805130, -2.5176010, 2.4762645
1: -10.2576609, -7.5326247, -10.2822790, -7.4614186, -2.2075682, 2.1721935
2: -6.3738594, -3.5874147, -6.4601684, -3.5581918, -2.2177930, 2.2749248
3: -2.4112382, 0.1215169, -2.4422724, 0.1332512, -1.7566361, 1.7851071
4: -7.0237675, -2.9460752, -7.0440612, -2.8905578, -3.1128664, 3.0877714
5: -8.9753265, -5.7503138, -8.9876585, -5.7355232, -2.3845186, 2.4026413
6: -19.4321270, -15.5613918, -19.4601593, -15.5480843, -3.1664400, 3.1504130
7: 4.2407608, 6.9734697, 4.2270689, 6.9874439, -2.7466831, 2.7464008
8: -7.1483359, -4.4544773, -7.1751137, -4.3977752, -2.3195982, 2.3168759
9: -7.1951995, -3.7835174, -7.2168636, -3.7630355, -2.6744480, 2.6551542

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1691936, upper bound: 1.1816091
time: 5.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1691954, upper bound: 1.1847035
time: 5.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044235, -13.5805264, -17.6044235, -13.5805130, -2.5289865, 2.4797375
1: -10.2822781, -7.4614248, -10.2822790, -7.4614182, -2.2202473, 2.1938720
2: -6.4601669, -3.5581925, -6.4601688, -3.5581913, -2.2427206, 2.2852745
3: -2.4422703, 0.1332499, -2.4422750, 0.1332523, -1.7687788, 1.7968373
4: -7.0440564, -2.8905563, -7.0440583, -2.8905554, -3.1207504, 3.1035910
5: -8.9876537, -5.7355223, -8.9876575, -5.7355251, -2.3903751, 2.4084694
6: -19.4601631, -15.5480938, -19.4601612, -15.5480843, -3.1841335, 3.1552649
7: 4.2270699, 6.9874420, 4.2270689, 6.9874458, -2.7603760, 2.7603731
8: -7.1751146, -4.3977766, -7.1751146, -4.3977733, -2.3304462, 2.3274987
9: -7.2168646, -3.7630396, -7.2168622, -3.7630348, -2.7055187, 2.6834230

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1691935, upper bound: 1.1794645
time: 5.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1691935, upper bound: 1.1825592
time: 5.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 25.97 seconds
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1694050, upper bound: 1.1794633
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1694049, upper bound: 1.1825593
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1694049, upper bound: 1.1794644
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1694049, upper bound: 1.1825592
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1681052, upper bound: 1.1810432
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1681049, upper bound: 1.1841437
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1691936, upper bound: 1.1816091
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1691954, upper bound: 1.1847035
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1691935, upper bound: 1.1794645
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 25.97
Output dim: 7, lower bound: -1.1691935, upper bound: 1.1825592

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5919838, -13.5854034, -17.5919838, -13.5854034, -2.4636459, 2.4636459
1: -10.2576609, -7.5326247, -10.2576609, -7.5326247, -2.1487889, 2.1487889
2: -6.3738594, -3.5874147, -6.3738594, -3.5874147, -2.2116990, 2.2116992
3: -2.4112382, 0.1215169, -2.4112382, 0.1215169, -1.7446232, 1.7446232
4: -7.0237675, -2.9460752, -7.0237675, -2.9460752, -3.0686102, 3.0686097
5: -8.9753265, -5.7503138, -8.9753265, -5.7503138, -2.3746958, 2.3746967
6: -19.4321270, -15.5613918, -19.4321270, -15.5613918, -3.1236496, 3.1236496
7: 4.2407608, 6.9734697, 4.2407608, 6.9734697, -2.7327089, 2.7327089
8: -7.1483359, -4.4544773, -7.1483359, -4.4544773, -2.2896519, 2.2896519
9: -7.1951995, -3.7835174, -7.1951995, -3.7835174, -2.6324468, 2.6324468

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1681294, upper bound: 1.1827635
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1693974, upper bound: 1.1796680
time: 8.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044235, -13.5805264, -17.5919838, -13.5854034, -2.4762640, 2.4683514
1: -10.2822781, -7.4614239, -10.2576609, -7.5326247, -2.1694601, 2.1850729
2: -6.4601665, -3.5581932, -6.3738594, -3.5874147, -2.2308674, 2.2162895
3: -2.4422700, 0.1332493, -2.4112382, 0.1215169, -1.7600007, 1.7549491
4: -7.0440555, -2.8905609, -7.0237675, -2.9460752, -3.0868158, 3.0983148
5: -8.9876518, -5.7355242, -8.9753265, -5.7503138, -2.3872151, 2.3838074
6: -19.4601593, -15.5480938, -19.4321270, -15.5613918, -3.1502104, 3.1375694
7: 4.2270689, 6.9874415, 4.2407608, 6.9734697, -2.7464008, 2.7466807
8: -7.1751122, -4.3977766, -7.1483359, -4.4544773, -2.3168740, 2.3172295
9: -7.2168627, -3.7630401, -7.1951995, -3.7835174, -2.6551533, 2.6523509

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1700247, upper bound: 1.1825510
time: 5.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1713315, upper bound: 1.1825517
time: 6.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5918236, -13.5858364, -17.5882168, -13.5901022, -2.4581475, 2.4587634
1: -10.2575569, -7.5326424, -10.2623711, -7.4767818, -2.1714411, 2.1592116
2: -6.3738203, -3.5874915, -6.4378462, -3.5996552, -2.2030087, 2.2117209
3: -2.4112363, 0.1214521, -2.4339993, 0.1182379, -1.7361131, 1.7493606
4: -7.0234480, -2.9461112, -6.9883142, -2.9186850, -3.0776281, 3.0693898
5: -8.9751596, -5.7503138, -8.9537287, -5.7457647, -2.3720717, 2.3497100
6: -19.4320297, -15.5614185, -19.4427795, -15.5620089, -3.1112852, 3.1238775
7: 4.2409711, 6.9734287, 4.2643256, 6.9667048, -2.7257338, 2.7091031
8: -7.1482363, -4.4545512, -7.1617804, -4.4029903, -2.3072429, 2.2977619
9: -7.1951523, -3.7840834, -7.2016153, -3.7783566, -2.6295667, 2.6307535

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1668466, upper bound: 1.1841374
time: 7.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1680972, upper bound: 1.1841366
time: 6.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5919838, -13.5854034, -17.6044235, -13.5805264, -2.4683514, 2.4762640
1: -10.2576609, -7.5326247, -10.2822781, -7.4614239, -2.1850727, 2.1694598
2: -6.3738594, -3.5874147, -6.4601665, -3.5581932, -2.2162895, 2.2308674
3: -2.4112382, 0.1215169, -2.4422700, 0.1332493, -1.7549491, 1.7600009
4: -7.0237675, -2.9460752, -7.0440555, -2.8905609, -3.0983148, 3.0868168
5: -8.9753265, -5.7503138, -8.9876518, -5.7355242, -2.3838077, 2.3872151
6: -19.4321270, -15.5613918, -19.4601593, -15.5480938, -3.1375694, 3.1502104
7: 4.2407608, 6.9734697, 4.2270689, 6.9874415, -2.7466807, 2.7464008
8: -7.1483359, -4.4544773, -7.1751122, -4.3977766, -2.3172297, 2.3168743
9: -7.1951995, -3.7835174, -7.2168627, -3.7630401, -2.6523509, 2.6551533

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1679199, upper bound: 1.1760372
time: 10.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1691860, upper bound: 1.1816015
time: 6.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044235, -13.5805264, -17.6044235, -13.5805264, -2.4797359, 2.4797359
1: -10.2822781, -7.4614248, -10.2822781, -7.4614248, -2.1938701, 2.1938696
2: -6.4601669, -3.5581925, -6.4601669, -3.5581925, -2.2412176, 2.2412174
3: -2.4422703, 0.1332499, -2.4422703, 0.1332499, -1.7687774, 1.7687776
4: -7.0440564, -2.8905563, -7.0440564, -2.8905563, -3.1035900, 3.1035891
5: -8.9876537, -5.7355223, -8.9876537, -5.7355223, -2.3903747, 2.3903751
6: -19.4601631, -15.5480938, -19.4601631, -15.5480938, -3.1552629, 3.1552639
7: 4.2270699, 6.9874420, 4.2270699, 6.9874420, -2.7603722, 2.7603722
8: -7.1751146, -4.3977766, -7.1751146, -4.3977766, -2.3274975, 2.3274975
9: -7.2168646, -3.7630396, -7.2168646, -3.7630396, -2.6834221, 2.6834221

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1677830, upper bound: 1.1810928
time: 5.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1700604, upper bound: 1.1810915
time: 5.94 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 26.25 seconds
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1681294, upper bound: 1.1827635
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1693974, upper bound: 1.1796680
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1700247, upper bound: 1.1825510
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1713315, upper bound: 1.1825517
IS_A2_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1668466, upper bound: 1.1841374
IS_A2_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1680972, upper bound: 1.1841366
IS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1679199, upper bound: 1.1760372
IS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1691860, upper bound: 1.1816015
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1677830, upper bound: 1.1810928
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 26.25
Output dim: 7, lower bound: -1.1700604, upper bound: 1.1810915

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17.5898666, -13.5860424, -17.5919838, -13.5854034, -2.4598942, 2.4609256
1: -10.2534084, -7.5342340, -10.2576609, -7.5326247, -2.1444445, 2.1472034
2: -6.3732238, -3.5911701, -6.3738594, -3.5874147, -2.2109780, 2.2079110
3: -2.4096231, 0.1161985, -2.4112382, 0.1215169, -1.7431965, 1.7395978
4: -7.0178890, -2.9475160, -7.0237675, -2.9460752, -3.0628262, 3.0671611
5: -8.9731655, -5.7508078, -8.9753265, -5.7503138, -2.3720865, 2.3734155
6: -19.4310951, -15.5627232, -19.4321270, -15.5613918, -3.1218090, 3.1214638
7: 4.2440329, 6.9725075, 4.2407608, 6.9734697, -2.7294369, 2.7317467
8: -7.1456423, -4.4575405, -7.1483359, -4.4544773, -2.2868929, 2.2865567
9: -7.1919827, -3.7865028, -7.1951995, -3.7835174, -2.6274304, 2.6272225

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1716387, upper bound: 1.1814948
time: 5.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1716387, upper bound: 1.1685840
time: 7.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.6023235, -13.5811539, -17.5919838, -13.5854034, -2.4725199, 2.4656339
1: -10.2780247, -7.4630218, -10.2576609, -7.5326247, -2.1651170, 2.1834574
2: -6.4595380, -3.5619507, -6.3738594, -3.5874147, -2.2301359, 2.2125025
3: -2.4406655, 0.1279321, -2.4112382, 0.1215169, -1.7586513, 1.7498581
4: -7.0381780, -2.8919902, -7.0237675, -2.9460752, -3.0810480, 3.0968366
5: -8.9854832, -5.7360191, -8.9753265, -5.7503138, -2.3845849, 2.3825271
6: -19.4591274, -15.5494347, -19.4321270, -15.5613918, -3.1484070, 3.1353645
7: 4.2303257, 6.9864626, 4.2407608, 6.9734697, -2.7431440, 2.7457018
8: -7.1724119, -4.4008350, -7.1483359, -4.4544773, -2.3141160, 2.3141026
9: -7.2135839, -3.7660251, -7.1951995, -3.7835174, -2.6500559, 2.6471272

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.1847181417998263, 1.1847155369154763]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2410.68 seconds
