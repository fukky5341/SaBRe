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
execution time: IAR + LP analysis = 15.22 + 33.21 = 48.42 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.58 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.7229785919189453
rel_dist={7: [-1.52484302938982, 1.5248425766176732]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.184716116461333, 1.184715436374983]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.7229785919189453
rel_dist={7: [-0.9113997367826396, 0.9113981821144339]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.7229785919189453
rel_dist={7: [-1.0527576805321806, 1.0527567151637864]}

## Binary Search Result
Binary search time: 204.91 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3346.67 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265152, upper bound: 1.6158858
time: 4.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265133, upper bound: 1.6265126
time: 4.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.88
Output dim: 7, lower bound: -1.6265152, upper bound: 1.6158858
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.88
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

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6158856
time: 5.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6158878
time: 6.06 seconds

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

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6247749, upper bound: 1.6193187
time: 7.02 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265122, upper bound: 1.6265111
time: 4.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.87
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6158856
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.87
Output dim: 7, lower bound: -1.6158858, upper bound: 1.6158878
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 26.87
Output dim: 7, lower bound: -1.6247749, upper bound: 1.6193187
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 26.87
Output dim: 7, lower bound: -1.6265122, upper bound: 1.6265111

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

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141337, upper bound: 1.6086855
time: 4.89 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158848, upper bound: 1.6158842
time: 5.00 seconds

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

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6086865, upper bound: 1.6141341
time: 5.13 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158845, upper bound: 1.6158867
time: 6.22 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -17.5871773, -13.6617861, -17.5956860, -13.6023941, -3.0993695, 3.0515904
1: -10.2652025, -7.5119863, -10.2636061, -7.4769664, -2.5078931, 2.4724884
2: -6.4010525, -3.5772104, -6.4439845, -3.5987794, -2.5871992, 2.6210887
3: -2.4022622, 0.1073880, -2.4296441, 0.1228554, -2.1070509, 2.1148469
4: -7.0244040, -2.9164214, -6.9907765, -2.9018035, -3.5393353, 3.5218511
5: -8.9563389, -5.7495213, -8.9540043, -5.7383347, -2.7789416, 2.7600002
6: -19.4457359, -15.5857649, -19.4446335, -15.5601606, -3.6916199, 3.6570649
7: 4.2370830, 6.9682693, 4.2611642, 6.9789515, -2.7418685, 2.7071052
8: -7.1522045, -4.4046659, -7.1657734, -4.4020939, -2.6691465, 2.6781352
9: -7.2076278, -3.7935030, -7.2090764, -3.7833521, -3.0444932, 3.0442424

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141354, upper bound: 1.6193190
time: 5.00 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141335, upper bound: 1.6193194
time: 16.43 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5972519, -13.5857944, -3.1403518, 3.0947268
1: -10.2822800, -7.4614229, -10.2654285, -7.4666867, -2.5357113, 2.5027590
2: -6.4601731, -3.5581913, -6.4559140, -3.5972736, -2.6075773, 2.6677752
3: -2.4422765, 0.1332507, -2.4377663, 0.1256859, -2.1263914, 2.1475511
4: -7.0440588, -2.8905525, -6.9938745, -2.8966470, -3.5677905, 3.5357952
5: -8.9876595, -5.7355232, -8.9602089, -5.7368917, -2.7976446, 2.7803907
6: -19.4601688, -15.5480900, -19.4462547, -15.5525570, -3.7134018, 3.6727433
7: 4.2270660, 6.9874468, 4.2598295, 6.9827995, -2.7557335, 2.7276173
8: -7.1751156, -4.3977699, -7.1687756, -4.4007754, -2.6979556, 2.6878066
9: -7.2168641, -3.7630351, -7.2100506, -3.7771640, -3.0615320, 3.0584946

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158847, upper bound: 1.6265110
time: 4.74 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158847, upper bound: 1.6265137
time: 6.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.49 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6141337, upper bound: 1.6086855
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6158848, upper bound: 1.6158842
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6086865, upper bound: 1.6141341
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6158845, upper bound: 1.6158867
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6141354, upper bound: 1.6193190
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6141335, upper bound: 1.6193194
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6158847, upper bound: 1.6265110
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 7, lower bound: -1.6158847, upper bound: 1.6265137

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -17.5709953, -13.6713810, -17.5866451, -13.6066780, -3.0751457, 3.0331256
1: -10.2452755, -7.5273380, -10.2605515, -7.4870501, -2.4767761, 2.4530382
2: -6.3787351, -3.6186585, -6.4259272, -3.6011555, -2.5621023, 2.5907478
3: -2.3939185, 0.0923653, -2.4258842, 0.1154133, -2.0847855, 2.0950892
4: -6.9687190, -2.9445286, -6.9852219, -2.9238317, -3.4880476, 3.4856877
5: -8.9223995, -5.7597561, -8.9475346, -5.7472057, -2.7307091, 2.7421942
6: -19.4283485, -15.5996542, -19.4411659, -15.5696039, -3.6523714, 3.6350818
7: 4.2743435, 6.9475808, 4.2656574, 6.9628687, -2.6885252, 2.6819234
8: -7.1388807, -4.4098778, -7.1587830, -4.4043007, -2.6455703, 2.6638043
9: -7.1923957, -3.8088126, -7.2006454, -3.7845335, -3.0245504, 3.0104632

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6092999, upper bound: 1.6086838
time: 5.18 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141330, upper bound: 1.6086833
time: 5.62 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.5882206, -13.5900984, -17.5882187, -13.5900822, -3.1161413, 3.0762460
1: -10.2623739, -7.4767785, -10.2623758, -7.4767718, -2.5046291, 2.4832830
2: -6.4378524, -3.5996547, -6.4378543, -3.5996532, -2.5824623, 2.6239834
3: -2.4340074, 0.1182399, -2.4340117, 0.1182419, -2.1041727, 2.1277664
4: -6.9883170, -2.9186780, -6.9883175, -2.9186773, -3.5131550, 3.4996319
5: -8.9537344, -5.7457666, -8.9537373, -5.7457619, -2.7494659, 2.7625842
6: -19.4427872, -15.5620098, -19.4427872, -15.5620022, -3.6741199, 3.6507320
7: 4.2643242, 6.9667082, 4.2643237, 6.9667125, -2.7023883, 2.7023845
8: -7.1617851, -4.4029846, -7.1617846, -4.4029832, -2.6743813, 2.6734369
9: -7.2016168, -3.7783523, -7.2016182, -3.7783484, -3.0415764, 3.0247040

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109252, upper bound: 1.6158818
time: 6.90 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158821, upper bound: 1.6158818
time: 5.17 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -17.5866451, -13.6066780, -17.5871773, -13.6617861, -3.0426965, 3.0919864
1: -10.2605515, -7.4870501, -10.2652025, -7.5119863, -2.4684176, 2.4975305
2: -6.4259272, -3.6011555, -6.4010525, -3.5772104, -2.6028395, 2.5844293
3: -2.4258842, 0.1154133, -2.4022622, 0.1073880, -2.1105223, 2.0929973
4: -6.9852219, -2.9238317, -7.0244040, -2.9164214, -3.5144243, 3.5167837
5: -8.9475346, -5.7472057, -8.9563389, -5.7495213, -2.7536125, 2.7684584
6: -19.4411659, -15.5696039, -19.4457359, -15.5857649, -3.6520424, 3.6697721
7: 4.2656574, 6.9628687, 4.2370830, 6.9682693, -2.7026119, 2.7257857
8: -7.1587830, -4.4043007, -7.1522045, -4.4046659, -2.6694679, 2.6596320
9: -7.2006454, -3.7845335, -7.2076278, -3.7935030, -3.0259867, 3.0408912

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193171, upper bound: 1.6092991
time: 5.68 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193171, upper bound: 1.6141305
time: 4.65 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.6044273, -13.5805225, -3.0858355, 3.1329665
1: -10.2623758, -7.4767718, -10.2822800, -7.4614229, -2.4986887, 2.5253468
2: -6.4378543, -3.5996532, -6.4601731, -3.5581913, -2.6495214, 2.6048031
3: -2.4340117, 0.1182419, -2.4422765, 0.1332507, -2.1432047, 2.1123381
4: -6.9883175, -2.9186773, -7.0440588, -2.8905525, -3.5283642, 3.5452337
5: -8.9537373, -5.7457619, -8.9876595, -5.7355232, -2.7740059, 2.7871614
6: -19.4427872, -15.5620022, -19.4601688, -15.5480900, -3.6677198, 3.6915522
7: 4.2643237, 6.9667125, 4.2270660, 6.9874468, -2.7231231, 2.7396464
8: -7.1617846, -4.4029832, -7.1751156, -4.3977699, -2.6791139, 2.6884673
9: -7.2016182, -3.7783484, -7.2168641, -3.7630351, -3.0402384, 3.0579295

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6147240, upper bound: 1.6133654
time: 7.42 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265088, upper bound: 1.6158828
time: 4.76 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -17.5871773, -13.6617861, -17.5866451, -13.6066780, -3.0919867, 3.0426970
1: -10.2652025, -7.5119863, -10.2605515, -7.4870501, -2.4975309, 2.4684176
2: -6.4010525, -3.5772104, -6.4259272, -3.6011555, -2.5844288, 2.6028395
3: -2.4022622, 0.1073880, -2.4258842, 0.1154133, -2.0929976, 2.1105225
4: -7.0244040, -2.9164214, -6.9852219, -2.9238317, -3.5167847, 3.5144238
5: -8.9563389, -5.7495213, -8.9475346, -5.7472057, -2.7684593, 2.7536120
6: -19.4457359, -15.5857649, -19.4411659, -15.5696039, -3.6697721, 3.6520433
7: 4.2370830, 6.9682693, 4.2656574, 6.9628687, -2.7257857, 2.7026119
8: -7.1522045, -4.4046659, -7.1587830, -4.4043007, -2.6596317, 2.6694679
9: -7.2076278, -3.7935030, -7.2006454, -3.7845335, -3.0408907, 3.0259871

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6092994, upper bound: 1.6193167
time: 5.31 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141309, upper bound: 1.6193163
time: 4.97 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5871773, -13.6617861, -17.6028519, -13.5970993, -3.1027181, 3.0607271
1: -10.2652025, -7.5119863, -10.2804623, -7.4716940, -2.5036039, 2.4798193
2: -6.4010525, -3.5772104, -6.4482465, -3.5596986, -2.6107841, 2.6253226
3: -2.4022622, 0.1073880, -2.4341698, 0.1304264, -2.1142659, 2.1228313
4: -7.0244040, -2.9164214, -7.0409527, -2.8957098, -3.5364962, 3.5340080
5: -8.9563389, -5.7495213, -8.9814644, -5.7369642, -2.7803488, 2.7917943
6: -19.4457359, -15.5857649, -19.4585476, -15.5556889, -3.6971636, 3.6798773
7: 4.2370830, 6.9682693, 4.2284021, 6.9835939, -2.7465110, 2.7398672
8: -7.1522045, -4.4046659, -7.1721134, -4.3990870, -2.6725550, 2.6907811
9: -7.2076278, -3.7935030, -7.2158885, -3.7692180, -3.0646763, 3.0505896

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6092994, upper bound: 1.6193173
time: 5.41 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141309, upper bound: 1.6193174
time: 7.38 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5882187, -13.5900822, -3.1329670, 3.0858345
1: -10.2822800, -7.4614229, -10.2623758, -7.4767718, -2.5253468, 2.4986882
2: -6.4601731, -3.5581913, -6.4378543, -3.5996532, -2.6048031, 2.6495214
3: -2.4422765, 0.1332507, -2.4340117, 0.1182419, -2.1123381, 2.1432047
4: -7.0440588, -2.8905525, -6.9883175, -2.9186773, -3.5452342, 3.5283642
5: -8.9876595, -5.7355232, -8.9537373, -5.7457619, -2.7871618, 2.7740054
6: -19.4601688, -15.5480900, -19.4427872, -15.5620022, -3.6915522, 3.6677198
7: 4.2270660, 6.9874468, 4.2643237, 6.9667125, -2.7396464, 2.7231231
8: -7.1751156, -4.3977699, -7.1617846, -4.4029832, -2.6884670, 2.6791136
9: -7.2168641, -3.7630351, -7.2016182, -3.7783484, -3.0579290, 3.0402384

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133658, upper bound: 1.6147235
time: 5.31 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158828, upper bound: 1.6265087
time: 5.21 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.6044312, -13.5805111, -3.1437645, 3.1038694
1: -10.2822800, -7.4614229, -10.2822809, -7.4614153, -2.5314283, 2.5100822
2: -6.4601731, -3.5581913, -6.4601760, -3.5581908, -2.6311617, 2.6720057
3: -2.4422765, 0.1332507, -2.4422810, 0.1332530, -2.1335998, 2.1572115
4: -7.0440588, -2.8905525, -7.0440617, -2.8905511, -3.5614414, 3.5479183
5: -8.9876595, -5.7355232, -8.9876633, -5.7355204, -2.7990541, 2.8121777
6: -19.4601688, -15.5480900, -19.4601688, -15.5480824, -3.7189474, 3.6955633
7: 4.2270660, 6.9874468, 4.2270651, 6.9874487, -2.7603827, 2.7603817
8: -7.1751156, -4.3977699, -7.1751165, -4.3977704, -2.7013655, 2.7004261
9: -7.2168641, -3.7630351, -7.2168632, -3.7630327, -3.0817165, 3.0648417

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109247, upper bound: 1.6265112
time: 5.51 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158818, upper bound: 1.6265112
time: 5.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.98 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6092999, upper bound: 1.6086838
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6141330, upper bound: 1.6086833
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6109252, upper bound: 1.6158818
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6158821, upper bound: 1.6158818
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6193171, upper bound: 1.6092991
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6193171, upper bound: 1.6141305
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6147240, upper bound: 1.6133654
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6265088, upper bound: 1.6158828
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6092994, upper bound: 1.6193167
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6141309, upper bound: 1.6193163
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6092994, upper bound: 1.6193173
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6141309, upper bound: 1.6193174
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6133658, upper bound: 1.6147235
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6158828, upper bound: 1.6265087
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6109247, upper bound: 1.6265112
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.98
Output dim: 7, lower bound: -1.6158818, upper bound: 1.6265112

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -17.5585556, -13.6728840, -17.5840321, -13.6083527, -3.0607572, 3.0283933
1: -10.2206879, -7.5979252, -10.2594738, -7.5057435, -2.4254360, 2.3813419
2: -6.2928286, -3.6478400, -6.4030895, -3.6021805, -2.4748716, 2.5131545
3: -2.3633814, 0.0807867, -2.4178238, 0.1143715, -2.0536566, 2.0694060
4: -6.9516149, -3.0000150, -6.9839373, -2.9382439, -3.4525971, 3.4287500
5: -8.9109421, -5.7745409, -8.9461308, -5.7510166, -2.7136559, 2.7257905
6: -19.4006233, -15.6129341, -19.4340248, -15.5705013, -3.6247826, 3.6068249
7: 4.2880573, 6.9342003, 4.2690377, 6.9611092, -2.6730518, 2.6651626
8: -7.1121664, -4.4665666, -7.1564317, -4.4193525, -2.5957355, 2.6046865
9: -7.1708150, -3.8281021, -7.1981931, -3.7899394, -2.9823580, 2.9880490

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6080101, upper bound: 1.6086725
time: 5.56 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6092852, upper bound: 1.6086687
time: 6.43 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -17.5709915, -13.6713810, -17.5866451, -13.6066780, -3.0741463, 3.0331244
1: -10.2452736, -7.5273404, -10.2605515, -7.4870501, -2.4702668, 2.4121995
2: -6.3787322, -3.6186581, -6.4259272, -3.6011555, -2.4953108, 2.5694218
3: -2.3939152, 0.0923645, -2.4258842, 0.1154133, -2.0700216, 2.0915203
4: -6.9687195, -2.9445279, -6.9852219, -2.9238317, -3.4859924, 3.4525046
5: -8.9223995, -5.7597570, -8.9475346, -5.7472057, -2.7307091, 2.7327909
6: -19.4283409, -15.5996599, -19.4411659, -15.5696039, -3.6486912, 3.6320109
7: 4.2743464, 6.9475813, 4.2656574, 6.9628687, -2.6885223, 2.6819239
8: -7.1388817, -4.4098821, -7.1587830, -4.4043007, -2.6420422, 2.6294575
9: -7.1923952, -3.8088150, -7.2006454, -3.7845335, -3.0187483, 3.0244155

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6127945, upper bound: 1.6086711
time: 5.77 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141158, upper bound: 1.6086681
time: 5.37 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949793, -17.5856018, -13.5917530, -3.1017070, 3.0687628
1: -10.2377253, -7.5479994, -10.2612934, -7.4954696, -2.4615211, 2.4111195
2: -6.3515325, -3.6289008, -6.4150143, -3.6006813, -2.4949150, 2.5597816
3: -2.4029112, 0.1065280, -2.4259477, 0.1171925, -2.0725889, 2.1077302
4: -6.9680362, -2.9742007, -6.9870405, -2.9330931, -3.4783401, 3.4426632
5: -8.9413948, -5.7605467, -8.9523335, -5.7495766, -2.7316685, 2.7461843
6: -19.4147167, -15.5752831, -19.4356461, -15.5628967, -3.6461430, 3.6224842
7: 4.2780342, 6.9527769, 4.2677011, 6.9649553, -2.6869211, 2.6850758
8: -7.1350536, -4.4596834, -7.1594329, -4.4180355, -2.6243730, 2.6142817
9: -7.1800356, -3.7988334, -7.1991644, -3.7837539, -2.9993863, 3.0013824

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6097076, upper bound: 1.6158705
time: 5.17 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109107, upper bound: 1.6158674
time: 5.58 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -17.5882149, -13.5901012, -17.5882187, -13.5900822, -3.1151366, 3.0762453
1: -10.2623749, -7.4767809, -10.2623758, -7.4767718, -2.5046291, 2.4424438
2: -6.4378490, -3.5996552, -6.4378543, -3.5996532, -2.5156713, 2.6161008
3: -2.4340060, 0.1182384, -2.4340117, 0.1182419, -2.0894094, 2.1277666
4: -6.9883165, -2.9186821, -6.9883175, -2.9186773, -3.5131531, 3.4664497
5: -8.9537334, -5.7457647, -8.9537373, -5.7457619, -2.7494650, 2.7531567
6: -19.4427795, -15.5620089, -19.4427872, -15.5620022, -3.6703844, 3.6476603
7: 4.2643242, 6.9667125, 4.2643237, 6.9667125, -2.7023883, 2.7023888
8: -7.1617837, -4.4029865, -7.1617846, -4.4029832, -2.6708479, 2.6390750
9: -7.2016153, -3.7783530, -7.2016182, -3.7783484, -3.0357695, 3.0386057

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6145972, upper bound: 1.6158702
time: 5.26 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158674, upper bound: 1.6158672
time: 5.00 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -17.5840321, -13.6083527, -17.5747490, -13.6636829, -3.0376449, 3.0776005
1: -10.2594738, -7.5057435, -10.2406502, -7.5826201, -2.3966994, 2.4338937
2: -6.4030895, -3.6021805, -6.3151135, -3.6063666, -2.5221181, 2.4971752
3: -2.4178238, 0.1143715, -2.3717303, 0.0957870, -2.0848384, 2.0618589
4: -6.9839373, -2.9382439, -7.0069308, -2.9719110, -3.4574718, 3.4678726
5: -8.9461308, -5.7510166, -8.9447956, -5.7643147, -2.7372022, 2.7513285
6: -19.4340248, -15.5705013, -19.4180088, -15.5990639, -3.6237707, 3.6421943
7: 4.2690377, 6.9611092, 4.2507772, 6.9547882, -2.6857505, 2.7103319
8: -7.1564317, -4.4193525, -7.1254377, -4.4613628, -2.6103368, 2.6097438
9: -7.1981931, -3.7899394, -7.1859670, -3.8129179, -3.0034752, 2.9986105

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193046, upper bound: 1.6080098
time: 5.20 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193018, upper bound: 1.6092845
time: 5.38 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -17.5866451, -13.6066780, -17.5871735, -13.6617889, -3.0426950, 3.0909863
1: -10.2605515, -7.4870501, -10.2652025, -7.5119867, -2.4275784, 2.4787354
2: -6.4259272, -3.6011555, -6.4010487, -3.5772107, -2.5783935, 2.5176382
3: -2.4258842, 0.1154133, -2.4022598, 0.1073871, -2.1069772, 2.0782340
4: -6.9852219, -2.9238317, -7.0244026, -2.9164233, -3.4812422, 3.5015898
5: -8.9475346, -5.7472057, -8.9563360, -5.7495222, -2.7442055, 2.7684574
6: -19.4411659, -15.5696039, -19.4457283, -15.5857668, -3.6489773, 3.6660953
7: 4.2656574, 6.9628687, 4.2370815, 6.9682679, -2.7026105, 2.7257872
8: -7.1587830, -4.4043007, -7.1522045, -4.4046688, -2.6351280, 2.6561081
9: -7.2006454, -3.7845335, -7.2076292, -3.7935042, -3.0399427, 3.0351133

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193046, upper bound: 1.6127939
time: 6.67 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6193018, upper bound: 1.6141152
time: 7.04 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -17.5870209, -13.5936260, -17.5949230, -13.6023130, -3.0623193, 3.1169147
1: -10.2606754, -7.4775252, -10.2715359, -7.4668283, -2.4915152, 2.5132403
2: -6.4331923, -3.6002374, -6.4314413, -3.5655024, -2.6240540, 2.5737586
3: -2.4256096, 0.1174879, -2.3909402, 0.1226578, -2.1094148, 2.0604835
4: -6.9875932, -2.9224417, -7.0373421, -2.9138517, -3.5046072, 3.5259757
5: -8.9450026, -5.7461138, -8.9339638, -5.7429266, -2.7583437, 2.7341161
6: -19.4419098, -15.5694780, -19.4508057, -15.5939751, -3.6211815, 3.6755762
7: 4.2652445, 6.9607038, 4.2357683, 6.9505405, -2.6852961, 2.7249355
8: -7.1604137, -4.4076838, -7.1629786, -4.4265733, -2.6487231, 2.6699533
9: -7.2007074, -3.7806563, -7.2099342, -3.7773788, -3.0248222, 3.0474563

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6147214, upper bound: 1.6085202
time: 9.48 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6147214, upper bound: 1.6133628
time: 11.25 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.6044273, -13.5805225, -3.0801201, 3.1321826
1: -10.2623758, -7.4767718, -10.2822781, -7.4614220, -2.5016832, 2.5253463
2: -6.4378543, -3.5996532, -6.4601707, -3.5581923, -2.6465335, 2.5934167
3: -2.4340117, 0.1182419, -2.4422727, 0.1332510, -2.1432042, 2.0779283
4: -6.9883175, -2.9186773, -7.0440578, -2.8905561, -3.5221634, 3.5435519
5: -8.9537373, -5.7457619, -8.9876537, -5.7355232, -2.7740059, 2.7514935
6: -19.4427872, -15.5620022, -19.4601688, -15.5480928, -3.6502218, 3.6915507
7: 4.2643237, 6.9667125, 4.2270670, 6.9874439, -2.7231202, 2.7396455
8: -7.1617846, -4.4029832, -7.1751146, -4.3977733, -2.6631160, 2.6881139
9: -7.2016182, -3.7783484, -7.2168655, -3.7630358, -3.0359602, 3.0579286

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6251992, upper bound: 1.6158710
time: 6.14 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6264937, upper bound: 1.6158688
time: 5.53 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -17.5747490, -13.6636829, -17.5840321, -13.6083527, -3.0776010, 3.0376446
1: -10.2406502, -7.5826201, -10.2594738, -7.5057435, -2.4338937, 2.3966994
2: -6.3151135, -3.6063666, -6.4030895, -3.6021805, -2.4971762, 2.5221181
3: -2.3717303, 0.0957870, -2.4178238, 0.1143715, -2.0618591, 2.0848386
4: -7.0069308, -2.9719110, -6.9839373, -2.9382439, -3.4678721, 3.4574718
5: -8.9447956, -5.7643147, -8.9461308, -5.7510166, -2.7513289, 2.7372026
6: -19.4180088, -15.5990639, -19.4340248, -15.5705013, -3.6421947, 3.6237717
7: 4.2507772, 6.9547882, 4.2690377, 6.9611092, -2.7103319, 2.6857505
8: -7.1254377, -4.4613628, -7.1564317, -4.4193525, -2.6097441, 2.6103365
9: -7.1859670, -3.8129179, -7.1981931, -3.7899394, -2.9986105, 3.0034747

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B1_A1_A1

### Relational analysis result of IS_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6080098, upper bound: 1.6193056
time: 5.60 seconds

## Relational analysis of IS_A2_A1_B1_A1_A2

### Relational analysis result of IS_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6092847, upper bound: 1.6193014
time: 5.50 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.5871735, -13.6617889, -17.5866451, -13.6066780, -3.0909863, 3.0426950
1: -10.2652025, -7.5119867, -10.2605515, -7.4870501, -2.4787354, 2.4275784
2: -6.4010487, -3.5772107, -6.4259272, -3.6011555, -2.5176382, 2.5783935
3: -2.4022598, 0.1073871, -2.4258842, 0.1154133, -2.0782337, 2.1069775
4: -7.0244026, -2.9164233, -6.9852219, -2.9238317, -3.5015898, 3.4812417
5: -8.9563360, -5.7495222, -8.9475346, -5.7472057, -2.7684574, 2.7442060
6: -19.4457283, -15.5857668, -19.4411659, -15.5696039, -3.6660948, 3.6489778
7: 4.2370815, 6.9682679, 4.2656574, 6.9628687, -2.7257872, 2.7026105
8: -7.1522045, -4.4046688, -7.1587830, -4.4043007, -2.6561084, 2.6351280
9: -7.2076292, -3.7935042, -7.2006454, -3.7845335, -3.0351133, 3.0399423

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6127942, upper bound: 1.6193040
time: 5.64 seconds

## Relational analysis of IS_A2_A1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6141155, upper bound: 1.6193010
time: 5.01 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17.5747490, -13.6636829, -17.6002369, -13.5987740, -3.0883288, 3.0556784
1: -10.2406502, -7.5826201, -10.2793932, -7.4903789, -2.4494152, 2.4081092
2: -6.3151135, -3.6063666, -6.4254088, -3.5607150, -2.5235357, 2.5446129
3: -2.3717303, 0.0957870, -2.4261212, 0.1293805, -2.0831223, 2.0939615
4: -7.0069308, -2.9719110, -7.0396652, -2.9101133, -3.4965611, 3.4770703
5: -8.9447956, -5.7643147, -8.9800587, -5.7407808, -2.7632184, 2.7753839
6: -19.4180088, -15.5990639, -19.4514141, -15.5565910, -3.6695824, 3.6516004
7: 4.2507772, 6.9547882, 4.2317824, 6.9818263, -2.7310491, 2.7230058
8: -7.1254377, -4.4613628, -7.1697469, -4.4141407, -2.6227565, 2.6316705
9: -7.1859670, -3.8129179, -7.2134209, -3.7746224, -3.0223970, 3.0280581

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087985, upper bound: 1.6193044
time: 6.98 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6100832, upper bound: 1.6193022
time: 6.49 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.5871735, -13.6617889, -17.6028519, -13.5970993, -3.1017189, 3.0607252
1: -10.2652025, -7.5119867, -10.2804623, -7.4716940, -2.4942381, 2.4389806
2: -6.4010487, -3.5772107, -6.4482465, -3.5596986, -2.5439925, 2.6008766
3: -2.4022598, 0.1073871, -2.4341698, 0.1304264, -2.0995021, 2.1160975
4: -7.0244026, -2.9164233, -7.0409527, -2.8957098, -3.5302610, 3.5008249
5: -8.9563360, -5.7495222, -8.9814644, -5.7369642, -2.7803488, 2.7823868
6: -19.4457283, -15.5857668, -19.4585476, -15.5556889, -3.6934586, 3.6768122
7: 4.2370815, 6.9682679, 4.2284021, 6.9835939, -2.7465124, 2.7398658
8: -7.1522045, -4.4046688, -7.1721134, -4.3990870, -2.6690316, 2.6564202
9: -7.2076292, -3.7935042, -7.2158885, -3.7692180, -3.0588980, 3.0644522

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6135694, upper bound: 1.6193051
time: 5.28 seconds

## Relational analysis of IS_A2_A1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6148884, upper bound: 1.6193018
time: 6.12 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.5949230, -13.6023130, -17.5870209, -13.5936260, -3.1169147, 3.0623188
1: -10.2715359, -7.4668283, -10.2606754, -7.4775252, -2.5132408, 2.4915152
2: -6.4314413, -3.5655024, -6.4331923, -3.6002374, -2.5737591, 2.6240542
3: -2.3909402, 0.1226578, -2.4256096, 0.1174879, -2.0604830, 2.1094146
4: -7.0373421, -2.9138517, -6.9875932, -2.9224417, -3.5259757, 3.5046072
5: -8.9339638, -5.7429266, -8.9450026, -5.7461138, -2.7341156, 2.7583442
6: -19.4508057, -15.5939751, -19.4419098, -15.5694780, -3.6755762, 3.6211815
7: 4.2357683, 6.9505405, 4.2652445, 6.9607038, -2.7249355, 2.6852961
8: -7.1629786, -4.4265733, -7.1604137, -4.4076838, -2.6699533, 2.6487231
9: -7.2099342, -3.7773788, -7.2007074, -3.7806563, -3.0474567, 3.0248227

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085227, upper bound: 1.6147210
time: 5.52 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133632, upper bound: 1.6147210
time: 5.84 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5882187, -13.5900822, -3.1321821, 3.0801201
1: -10.2822781, -7.4614220, -10.2623758, -7.4767718, -2.5253468, 2.5016837
2: -6.4601707, -3.5581923, -6.4378543, -3.5996532, -2.5934162, 2.6465335
3: -2.4422727, 0.1332510, -2.4340117, 0.1182419, -2.0779281, 2.1432042
4: -7.0440578, -2.8905561, -6.9883175, -2.9186773, -3.5435529, 3.5221643
5: -8.9876537, -5.7355232, -8.9537373, -5.7457619, -2.7514935, 2.7740054
6: -19.4601688, -15.5480928, -19.4427872, -15.5620022, -3.6915512, 3.6502223
7: 4.2270670, 6.9874439, 4.2643237, 6.9667125, -2.7396455, 2.7231202
8: -7.1751146, -4.3977733, -7.1617846, -4.4029832, -2.6881137, 2.6631162
9: -7.2168655, -3.7630358, -7.2016182, -3.7783484, -3.0579290, 3.0359597

Time for backsubstitution: 14.55 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.7229785919189453
rel_dist={7: [-1.6265317094795417, 1.626531505248428]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100405, upper bound: 1.3032396
time: 5.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105111, upper bound: 1.3105086
time: 5.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.71
Output dim: 7, lower bound: -1.3100405, upper bound: 1.3032396
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.71
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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032399, upper bound: 1.3032397
time: 5.24 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032421, upper bound: 1.3032408
time: 5.88 seconds

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083398, upper bound: 1.3051844
time: 6.36 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105077, upper bound: 1.3105063
time: 5.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 7, lower bound: -1.3032399, upper bound: 1.3032397
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 7, lower bound: -1.3032421, upper bound: 1.3032408
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 7, lower bound: -1.3083398, upper bound: 1.3051844
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 7, lower bound: -1.3105077, upper bound: 1.3105063

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3010929, upper bound: 1.2979363
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032409, upper bound: 1.3032400
time: 16.04 seconds

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

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6209

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2979371, upper bound: 1.3010923
time: 5.66 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032385, upper bound: 1.3032387
time: 6.57 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -17.5871773, -13.6617861, -17.5945282, -13.6144810, -2.6325848, 2.5980361
1: -10.2652025, -7.5119863, -10.2622633, -7.4844508, -2.3005414, 2.2735953
2: -6.4010525, -3.5772104, -6.4353147, -3.5998945, -2.3783607, 2.4015157
3: -2.4022622, 0.1073880, -2.4237337, 0.1207376, -1.8816857, 1.8855226
4: -7.0244040, -2.9164214, -6.9884963, -2.9055591, -3.2473040, 3.2255650
5: -8.9563389, -5.7495213, -8.9495010, -5.7394352, -2.5180087, 2.4942107
6: -19.4457359, -15.5857649, -19.4434338, -15.5657005, -3.3103342, 3.2811675
7: 4.2370830, 6.9682693, 4.2621584, 6.9761500, -2.7390671, 2.7061110
8: -7.1522045, -4.4046659, -7.1635180, -4.4030542, -2.4427471, 2.4516633
9: -7.2076278, -3.7935030, -7.2083554, -3.7878532, -2.7602301, 2.7632570

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083403, upper bound: 1.3023374
time: 5.44 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083383, upper bound: 1.3051829
time: 5.86 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5972519, -13.5858002, -2.6875496, 2.6354172
1: -10.2822800, -7.4614229, -10.2654266, -7.4666901, -2.3398781, 2.3015113
2: -6.4601731, -3.5581913, -6.4559107, -3.5972733, -2.3924451, 2.4566123
3: -2.4422765, 0.1332507, -2.4377654, 0.1256843, -1.8983326, 1.9244831
4: -7.0440588, -2.8905525, -6.9938736, -2.8966587, -3.2791262, 3.2390380
5: -8.9876595, -5.7355232, -8.9602051, -5.7368975, -2.5335331, 2.5202165
6: -19.4601688, -15.5480900, -19.4462528, -15.5525637, -3.3375969, 3.2938375
7: 4.2270660, 6.9874468, 4.2598314, 6.9827957, -2.7557297, 2.7276154
8: -7.1751156, -4.3977699, -7.1687756, -4.4007754, -2.4726067, 2.4626527
9: -7.2168641, -3.7630351, -7.2100439, -3.7771654, -2.7822132, 2.7744598

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105062, upper bound: 1.3076524
time: 5.13 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105062, upper bound: 1.3105049
time: 5.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.30 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3010929, upper bound: 1.2979363
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3032409, upper bound: 1.3032400
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.2979371, upper bound: 1.3010923
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3032385, upper bound: 1.3032387
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3083403, upper bound: 1.3023374
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3083383, upper bound: 1.3051829
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3105062, upper bound: 1.3076524
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.30
Output dim: 7, lower bound: -1.3105062, upper bound: 1.3105049

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -17.5709953, -13.6713810, -17.5854988, -13.6187592, -2.6088865, 2.5795729
1: -10.2452755, -7.5273380, -10.2592134, -7.4945326, -2.2736025, 2.2557917
2: -6.3787351, -3.6186585, -6.4172592, -3.6022701, -2.3560944, 2.3742971
3: -2.3939185, 0.0923653, -2.4199753, 0.1132981, -1.8599815, 1.8657637
4: -6.9687190, -2.9445286, -6.9829464, -2.9275718, -3.1969709, 3.1961470
5: -8.9223995, -5.7597561, -8.9430332, -5.7483034, -2.4698257, 2.4764061
6: -19.4283485, -15.5996542, -19.4399605, -15.5751410, -3.2720995, 3.2591848
7: 4.2743435, 6.9475808, 4.2666492, 6.9600754, -2.6857319, 2.6809316
8: -7.1388807, -4.4098778, -7.1565313, -4.4052596, -2.4208803, 2.4373333
9: -7.1923957, -3.8088126, -7.1999278, -3.7890363, -2.7402916, 2.7302823

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2982431, upper bound: 1.2979353
time: 5.35 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3010916, upper bound: 1.2979352
time: 5.06 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.5882206, -13.5900984, -17.5882187, -13.5900822, -2.6638603, 2.6169403
1: -10.2623739, -7.4767785, -10.2623758, -7.4767718, -2.3087988, 2.2836795
2: -6.4378524, -3.5996547, -6.4378543, -3.5996532, -2.3701582, 2.4193625
3: -2.4340074, 0.1182399, -2.4340117, 0.1182419, -1.8766770, 1.9047005
4: -6.9883170, -2.9186780, -6.9883175, -2.9186773, -3.2258701, 3.2096219
5: -8.9537344, -5.7457666, -8.9537373, -5.7457619, -2.4854031, 2.5024118
6: -19.4427872, -15.5620098, -19.4427872, -15.5620022, -3.2993298, 3.2718287
7: 4.2643242, 6.9667082, 4.2643237, 6.9667125, -2.7023883, 2.7023845
8: -7.1617851, -4.4029846, -7.1617846, -4.4029832, -2.4507370, 2.4482856
9: -7.2016168, -3.7783523, -7.2016182, -3.7783484, -2.7622604, 2.7414727

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003791, upper bound: 1.3032386
time: 5.52 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032397, upper bound: 1.3032390
time: 5.45 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -17.5854988, -13.6187592, -17.5871773, -13.6617861, -2.5891442, 2.6243086
1: -10.2592134, -7.4945326, -10.2652025, -7.5119863, -2.2711711, 2.2900910
2: -6.4172592, -3.6022701, -6.4010525, -3.5772104, -2.3832679, 2.3784213
3: -2.4199753, 0.1132981, -2.4022622, 0.1073880, -1.8807530, 1.8681936
4: -6.9829464, -2.9275718, -7.0244040, -2.9164214, -3.2248840, 3.2247567
5: -8.9430332, -5.7483034, -8.9563389, -5.7495213, -2.4878244, 2.5075731
6: -19.4399605, -15.5751410, -19.4457359, -15.5857649, -3.2761459, 3.2895002
7: 4.2666492, 6.9600754, 4.2370830, 6.9682693, -2.7016201, 2.7229924
8: -7.1565313, -4.4052596, -7.1522045, -4.4046659, -2.4429970, 2.4349420
9: -7.1999278, -3.7890363, -7.2076278, -3.7935030, -2.7458062, 2.7566323

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047214, upper bound: 1.2982401
time: 5.53 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047233, upper bound: 1.3010901
time: 5.31 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900822, -17.6044273, -13.5805225, -2.6265287, 2.6806855
1: -10.2623758, -7.4767718, -10.2822800, -7.4614229, -2.2990847, 2.3295164
2: -6.4378543, -3.5996532, -6.4601731, -3.5581913, -2.4383607, 2.3924994
3: -2.4340117, 0.1182419, -2.4422765, 0.1332507, -1.9201388, 1.8848426
4: -6.9883175, -2.9186773, -7.0440588, -2.8905525, -3.2383537, 3.2565713
5: -8.9537373, -5.7457619, -8.9876595, -5.7355232, -2.5138340, 2.5230980
6: -19.4427872, -15.5620022, -19.4601688, -15.5480900, -3.2888165, 3.3167620
7: 4.2643237, 6.9667125, 4.2270660, 6.9874468, -2.7231231, 2.7396464
8: -7.1617846, -4.4029832, -7.1751156, -4.3977699, -2.4539623, 2.4648230
9: -7.2016182, -3.7783484, -7.2168641, -3.7630351, -2.7570066, 2.7786136

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100367, upper bound: 1.3003783
time: 5.15 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100387, upper bound: 1.3032375
time: 5.51 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -17.5828342, -13.6635141, -17.5820942, -13.6193867, -2.6158164, 2.5832686
1: -10.2634449, -7.5428381, -10.2376575, -7.5556469, -2.2277493, 2.2183318
2: -6.3632364, -3.5788851, -6.3490038, -3.6291034, -2.3037176, 2.3133099
3: -2.3890686, 0.1056831, -2.3926978, 0.1090789, -1.8565559, 1.8530703
4: -7.0232544, -2.9403460, -6.9681511, -2.9610496, -3.1903720, 3.1811323
5: -8.9542828, -5.7558556, -8.9371490, -5.7542210, -2.5009379, 2.4732733
6: -19.4339867, -15.5872707, -19.4153748, -15.5789909, -3.2746849, 3.2524076
7: 4.2426953, 6.9654880, 4.2758398, 6.9622197, -2.7195244, 2.6896482
8: -7.1482525, -4.4296865, -7.1368160, -4.4597497, -2.3818426, 2.3891289
9: -7.2035122, -3.8021045, -7.1867223, -3.8083298, -2.7345285, 2.7209778

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047910, upper bound: 1.2933979
time: 5.33 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083392, upper bound: 1.3023350
time: 5.08 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -17.5871792, -13.6617908, -17.5945244, -13.6144810, -2.6293678, 2.5968566
1: -10.2652025, -7.5119858, -10.2622633, -7.4844522, -2.2507515, 2.2735944
2: -6.4010515, -3.5772114, -6.4353104, -3.5998964, -2.3737593, 2.3219976
3: -2.4022613, 0.1073885, -2.4237313, 0.1207366, -1.8816853, 1.8681548
4: -7.0244031, -2.9164224, -6.9884953, -2.9055610, -3.2070742, 3.2255635
5: -8.9563370, -5.7495213, -8.9495010, -5.7394371, -2.5063004, 2.4942098
6: -19.4457340, -15.5857658, -19.4434261, -15.5657043, -3.3057942, 3.2711315
7: 4.2370806, 6.9682703, 4.2621603, 6.9761496, -2.7390690, 2.7061100
8: -7.1522036, -4.4046655, -7.1635170, -4.4030557, -2.3971720, 2.4481199
9: -7.2076297, -3.7935030, -7.2083549, -3.7878559, -2.7656097, 2.7574525

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083316, upper bound: 1.3039149
time: 5.02 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083288, upper bound: 1.3051725
time: 5.27 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6000881, -13.5833073, -17.5848045, -13.5906773, -2.6782503, 2.6197753
1: -10.2804890, -7.4924788, -10.2407990, -7.5379009, -2.2670550, 2.2460346
2: -6.4222226, -3.5598984, -6.3695879, -3.6265078, -2.3173079, 2.3683536
3: -2.4288986, 0.1314930, -2.4066978, 0.1139760, -1.8730426, 1.8919685
4: -7.0419259, -2.9144955, -6.9735951, -2.9521682, -3.2213154, 3.1946621
5: -8.9853125, -5.7418566, -8.9478683, -5.7516832, -2.5162339, 2.4993072
6: -19.4483147, -15.5495958, -19.4181862, -15.5658455, -3.3018513, 3.2650642
7: 4.2326822, 6.9844866, 4.2735119, 6.9688697, -2.7361875, 2.7109747
8: -7.1711521, -4.4227948, -7.1420622, -4.4574766, -2.4116230, 2.3999481
9: -7.2127428, -3.7720156, -7.1884146, -3.7976503, -2.7565155, 2.7318664

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3074640, upper bound: 1.2992586
time: 5.70 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105044, upper bound: 1.3076495
time: 5.56 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5972462, -13.5857992, -2.6875477, 2.6342382
1: -10.2822790, -7.4614220, -10.2654276, -7.4666920, -2.2918239, 2.3015089
2: -6.4601727, -3.5581918, -6.4559078, -3.5972743, -2.3874812, 2.3770912
3: -2.4422760, 0.1332504, -2.4377630, 0.1256837, -1.8983307, 1.9071155
4: -7.0440593, -2.8905525, -6.9938726, -2.8966634, -3.2389059, 3.2390366
5: -8.9876595, -5.7355232, -8.9602051, -5.7368975, -2.5218163, 2.5202150
6: -19.4601669, -15.5480900, -19.4462452, -15.5525627, -3.3345318, 3.2837858
7: 4.2270679, 6.9874468, 4.2598324, 6.9827957, -2.7557278, 2.7276144
8: -7.1751165, -4.3977709, -7.1687727, -4.4007778, -2.4270291, 2.4591084
9: -7.2168632, -3.7630367, -7.2100453, -3.7771678, -2.7875896, 2.7686543

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3104996, upper bound: 1.3092711
time: 5.18 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3104967, upper bound: 1.3104951
time: 5.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.52 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.2982431, upper bound: 1.2979353
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3010916, upper bound: 1.2979352
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3003791, upper bound: 1.3032386
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3032397, upper bound: 1.3032390
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3047214, upper bound: 1.2982401
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3047233, upper bound: 1.3010901
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3100367, upper bound: 1.3003783
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3100387, upper bound: 1.3032375
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3047910, upper bound: 1.2933979
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3083392, upper bound: 1.3023350
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3083316, upper bound: 1.3039149
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3083288, upper bound: 1.3051725
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3074640, upper bound: 1.2992586
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3105044, upper bound: 1.3076495
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3104996, upper bound: 1.3092711
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.52
Output dim: 7, lower bound: -1.3104967, upper bound: 1.3104951

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -17.5585556, -13.6728840, -17.5811577, -13.6215515, -2.5932951, 2.5730200
1: -10.2206879, -7.5979252, -10.2574148, -7.5255961, -2.2057099, 2.1834221
2: -6.2928286, -3.6478400, -6.3793106, -3.6039748, -2.2682405, 2.2797167
3: -2.3633814, 0.0807867, -2.4065905, 0.1115673, -1.8280725, 1.8331387
4: -6.9516149, -3.0000150, -6.9807897, -2.9515250, -3.1518221, 3.1383638
5: -8.9109421, -5.7745409, -8.9406853, -5.7546329, -2.4496646, 2.4590716
6: -19.4006233, -15.6129341, -19.4281025, -15.5766401, -3.2437172, 3.2234449
7: 4.2880573, 6.9342003, 4.2722607, 6.9571319, -2.6690745, 2.6619396
8: -7.1121664, -4.4665666, -7.1525970, -4.4302778, -2.3583274, 2.3764045
9: -7.1708150, -3.8281021, -7.1958361, -3.7980134, -2.6976819, 2.7055721

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2947268, upper bound: 1.2890111
time: 4.99 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2982421, upper bound: 1.2979333
time: 5.36 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -17.5709915, -13.6713810, -17.5854969, -13.6187611, -2.6069989, 2.5795717
1: -10.2452736, -7.5273404, -10.2592115, -7.4945321, -2.2618079, 2.2077374
2: -6.3787322, -3.6186581, -6.4172564, -3.6022706, -2.2774973, 2.3498504
3: -2.3939152, 0.0923645, -2.4199743, 0.1132985, -1.8426137, 1.8601229
4: -6.9687195, -2.9445279, -6.9829459, -2.9275701, -3.1939635, 3.1571040
5: -8.9223995, -5.7597570, -8.9430332, -5.7483015, -2.4698248, 2.4647198
6: -19.4283409, -15.5996599, -19.4399605, -15.5751371, -3.2621317, 3.2561131
7: 4.2743464, 6.9475813, 4.2666483, 6.9600744, -2.6857281, 2.6809330
8: -7.1388817, -4.4098821, -7.1565313, -4.4052601, -2.4173517, 2.3917959
9: -7.1923952, -3.8088150, -7.1999283, -3.7890356, -2.7344871, 2.7358065

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2998566, upper bound: 1.2979264
time: 6.02 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3010833, upper bound: 1.2979254
time: 5.35 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -17.5757713, -13.5949793, -17.5838776, -13.5928688, -2.6482096, 2.6076329
1: -10.2377253, -7.5479994, -10.2605705, -7.5078421, -2.2501888, 2.2108364
2: -6.3515325, -3.6289008, -6.3999004, -3.6013699, -2.2819781, 2.3347464
3: -2.4029112, 0.1065280, -2.4206142, 0.1164905, -1.8443089, 1.8776073
4: -6.9680362, -2.9742007, -6.9861903, -2.9426339, -3.1809292, 3.1518202
5: -8.9413948, -5.7605467, -8.9513941, -5.7520905, -2.4645042, 2.4850845
6: -19.4147167, -15.5752831, -19.4309235, -15.5635033, -3.2705593, 3.2360992
7: 4.2780342, 6.9527769, 4.2699337, 6.9637675, -2.6857333, 2.6828432
8: -7.1350536, -4.4596834, -7.1578469, -4.4280062, -2.3880062, 2.3872662
9: -7.1800356, -3.7988334, -7.1975260, -3.7873292, -2.7196522, 2.7158556

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2973462, upper bound: 1.2948392
time: 5.20 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003773, upper bound: 1.3032343
time: 5.21 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -17.5882149, -13.5901012, -17.5882187, -13.5900850, -2.6626825, 2.6169384
1: -10.2623749, -7.4767809, -10.2623739, -7.4767733, -2.3063593, 2.2356234
2: -6.4378490, -3.5996552, -6.4378533, -3.5996532, -2.2915635, 2.4049401
3: -2.4340060, 0.1182384, -2.4340105, 0.1182411, -1.8593092, 1.9046700
4: -6.9883165, -2.9186821, -6.9883199, -2.9186759, -3.2258682, 3.1705785
5: -8.9537334, -5.7457647, -8.9537363, -5.7457628, -2.4854021, 2.4907007
6: -19.4427795, -15.5620089, -19.4427872, -15.5620022, -3.2893066, 3.2687554
7: 4.2643242, 6.9667125, 4.2643242, 6.9667125, -2.7023883, 2.7023883
8: -7.1617837, -4.4029865, -7.1617842, -4.4029841, -2.4472017, 2.4027300
9: -7.2016153, -3.7783530, -7.2016163, -3.7783475, -2.7564516, 2.7469492

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 478
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3020280, upper bound: 1.3032294
time: 5.62 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3032269, upper bound: 1.3032282
time: 5.85 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -17.5811577, -13.6215515, -17.5747490, -13.6636829, -2.5822711, 2.6087246
1: -10.2574148, -7.5255961, -10.2406502, -7.5826201, -2.1987796, 2.2141676
2: -6.3793106, -3.6039748, -6.3151135, -3.6063666, -2.2886801, 2.2905436
3: -2.4065905, 0.1115673, -2.3717303, 0.0957870, -1.8470159, 1.8362753
4: -6.9807897, -2.9515250, -7.0069308, -2.9719110, -3.1670856, 3.1670985
5: -8.9406853, -5.7546329, -8.9447956, -5.7643147, -2.4704843, 2.4873381
6: -19.4281025, -15.5766401, -19.4180088, -15.5990639, -3.2403908, 3.2611279
7: 4.2722607, 6.9571319, 4.2507772, 6.9547882, -2.6825275, 2.7063546
8: -7.1525970, -4.4302778, -7.1254377, -4.4613628, -2.3820548, 2.3723357
9: -7.1958361, -3.7980134, -7.1859670, -3.8129179, -2.7209973, 2.7139354

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2957927, upper bound: 1.2947244
time: 5.51 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047209, upper bound: 1.2982393
time: 5.82 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -17.5854969, -13.6187611, -17.5871735, -13.6617889, -2.5891418, 2.6224210
1: -10.2592115, -7.4945321, -10.2652025, -7.5119867, -2.2231164, 2.2702765
2: -6.4172564, -3.6022706, -6.4010487, -3.5772107, -2.3588219, 2.2998252
3: -2.4199743, 0.1132985, -2.4022598, 0.1073871, -1.8740194, 1.8508260
4: -6.9829459, -2.9275701, -7.0244026, -2.9164233, -3.1858416, 3.2095623
5: -8.9430332, -5.7483015, -8.9563360, -5.7495222, -2.4761353, 2.5075717
6: -19.4399605, -15.5751371, -19.4457283, -15.5857668, -3.2730799, 3.2795353
7: 4.2666483, 6.9600744, 4.2370815, 6.9682679, -2.7016196, 2.7229929
8: -7.1565313, -4.4052601, -7.1522045, -4.4046688, -2.3974667, 2.4314182
9: -7.1999283, -3.7890356, -7.2076292, -3.7935042, -2.7513332, 2.7508526

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047145, upper bound: 1.2998560
time: 9.07 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047118, upper bound: 1.3010804
time: 12.78 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -17.5838776, -13.5928688, -17.5919838, -13.5853996, -2.6172147, 2.6650391
1: -10.2605705, -7.5078421, -10.2576599, -7.5326257, -2.2262344, 2.2586379
2: -6.3999004, -3.6013699, -6.3738627, -3.5874131, -2.3437042, 2.3043261
3: -2.4206142, 0.1164905, -2.4112420, 0.1215165, -1.8914938, 1.8525093
4: -6.9861903, -2.9426339, -7.0237684, -2.9460735, -3.1805687, 3.1964526
5: -8.9513941, -5.7520905, -8.9753323, -5.7503128, -2.4965010, 2.5022144
6: -19.4309235, -15.5635033, -19.4321289, -15.5613880, -3.2530727, 3.2880378
7: 4.2699337, 6.9637675, 4.2407618, 6.9734726, -2.7035389, 2.7230058
8: -7.1578469, -4.4280062, -7.1483364, -4.4544759, -2.3929291, 2.4020731
9: -7.1975260, -3.7873292, -7.1951985, -3.7835174, -2.7313900, 2.7359166

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3016436, upper bound: 1.2973460
time: 5.46 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100343, upper bound: 1.3003762
time: 5.71 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -17.5882187, -13.5900850, -17.6044235, -13.5805225, -2.6265268, 2.6795068
1: -10.2623739, -7.4767733, -10.2822800, -7.4614244, -2.2510290, 2.3148189
2: -6.4378533, -3.5996532, -6.4601693, -3.5581927, -2.4139042, 2.3139043
3: -2.4340105, 0.1182411, -2.4422758, 0.1332505, -1.9185717, 1.8674750
4: -6.9883199, -2.9186759, -7.0440578, -2.8905540, -3.1993103, 3.2414308
5: -8.9537363, -5.7457628, -8.9876585, -5.7355223, -2.5021195, 2.5230970
6: -19.4427872, -15.5620022, -19.4601631, -15.5480900, -3.2857513, 3.3067436
7: 4.2643242, 6.9667125, 4.2270689, 6.9874458, -2.7231216, 2.7396436
8: -7.1617842, -4.4029841, -7.1751146, -4.3977733, -2.4084148, 2.4612930
9: -7.2016163, -3.7783475, -7.2168636, -3.7630379, -2.7624860, 2.7728286

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100277, upper bound: 1.3020252
time: 5.51 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3100269, upper bound: 1.3032261
time: 5.35 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -17.5731316, -13.6852713, -17.5799351, -13.6257095, -2.5875514, 2.5585876
1: -10.2526617, -7.5482473, -10.2346306, -7.5569925, -2.2150393, 2.2094479
2: -6.3346338, -3.5862896, -6.3407078, -3.6301470, -2.2720995, 2.2836962
3: -2.3379412, 0.0948336, -2.3777471, 0.1077144, -1.8038893, 1.8042088
4: -7.0165420, -2.9635637, -6.9668503, -2.9677558, -3.1680145, 3.1569157
5: -8.9007664, -5.7634506, -8.9215908, -5.7548704, -2.4476757, 2.4428077
6: -19.4244823, -15.6330509, -19.4138050, -15.5923090, -3.2380219, 3.2054548
7: 4.2513828, 6.9288034, 4.2774820, 6.9515104, -2.7001276, 2.6513214
8: -7.1357322, -4.4584708, -7.1343155, -4.4681234, -2.3588238, 2.3574395
9: -7.1965227, -3.8163185, -7.1850824, -3.8124328, -2.7221780, 2.7046819

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047839, upper bound: 1.2921559
time: 5.57 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047830, upper bound: 1.2933879
time: 5.59 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.5828342, -13.6635170, -17.5820942, -13.6193867, -2.6135309, 2.5744495
1: -10.2634430, -7.5428381, -10.2376575, -7.5556469, -2.2277484, 2.2210150
2: -6.3632336, -3.5788870, -6.3490038, -3.6291034, -2.2903228, 2.3103113
3: -2.3890629, 0.1056840, -2.3926978, 0.1090789, -1.8160782, 1.8492944
4: -7.0232558, -2.9403489, -6.9681511, -2.9610496, -3.1886802, 3.1738415
5: -8.9542789, -5.7558565, -8.9371490, -5.7542210, -2.4589868, 2.4732738
6: -19.4339828, -15.5872717, -19.4153748, -15.5789909, -3.2746868, 3.2318234
7: 4.2426968, 6.9654846, 4.2758398, 6.9622197, -2.7195230, 2.6896448
8: -7.1482525, -4.4296885, -7.1368160, -4.4597497, -2.3815246, 2.3691552
9: -7.2035141, -3.8021071, -7.1867223, -3.8083298, -2.7345281, 2.7159429

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083305, upper bound: 1.3011042
time: 5.23 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083296, upper bound: 1.3023259
time: 5.21 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -17.5871792, -13.6617908, -17.5924263, -13.6151104, -2.6266503, 2.5931039
1: -10.2652025, -7.5119858, -10.2580156, -7.4860449, -2.2492008, 2.2692528
2: -6.4010515, -3.5772114, -6.4346776, -3.6036487, -2.3699751, 2.3212833
3: -2.4022613, 0.1073885, -2.4221108, 0.1154207, -1.8766713, 1.8666968
4: -7.0244031, -2.9164224, -6.9826388, -2.9069691, -3.2056417, 3.2197981
5: -8.9563370, -5.7495213, -8.9473591, -5.7399321, -2.5050216, 2.4915729
6: -19.4457340, -15.5857658, -19.4423981, -15.5670366, -3.3035941, 3.2692947
7: 4.2370806, 6.9682703, 4.2654505, 6.9751730, -2.7380924, 2.7028198
8: -7.1522036, -4.4046655, -7.1608100, -4.4061136, -2.3940773, 2.4453623
9: -7.2076297, -3.7935030, -7.2050772, -3.7908330, -2.7603936, 2.7523651

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047839, upper bound: 1.2949070
time: 5.37 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083286, upper bound: 1.3039124
time: 5.25 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.5871696, -13.6617880, -17.6021538, -13.6088209, -2.6305642, 2.6229701
1: -10.2651854, -7.5119910, -10.2677612, -7.4308853, -2.2683644, 2.2802830
2: -6.4010491, -3.5772195, -6.4630651, -3.5935678, -2.3806100, 2.3285725
3: -2.4022555, 0.1073706, -2.4629183, 0.1281664, -1.8930569, 1.8762429
4: -7.0243812, -2.9164264, -6.9980545, -2.8521895, -3.2226496, 3.2356348
5: -8.9563322, -5.7495251, -8.9600868, -5.7357302, -2.5148649, 2.5059595
6: -19.4457302, -15.5857687, -19.4500237, -15.5597343, -3.3123207, 3.2772365
7: 4.2370892, 6.9682674, 4.2479095, 6.9944248, -2.7573357, 2.7203579
8: -7.1521974, -4.4046760, -7.2102003, -4.3999977, -2.4016190, 2.4703095
9: -7.2076178, -3.7935123, -7.2611494, -3.7850976, -2.7663040, 2.8035038

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3047830, upper bound: 1.2962211
time: 5.50 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3083277, upper bound: 1.3051703
time: 5.27 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.5905819, -13.6050987, -17.5826683, -13.5969982, -2.6503143, 2.5951037
1: -10.2697382, -7.4978933, -10.2377720, -7.5392480, -2.2544317, 2.2371464
2: -6.3934932, -3.5671961, -6.3612761, -3.6275396, -2.2857442, 2.3387766
3: -2.3775682, 0.1209023, -2.3917298, 0.1126418, -1.8204761, 1.8485863
4: -7.0352077, -2.9377866, -6.9723005, -2.9588809, -3.1989317, 3.1704469
5: -8.9316158, -5.7492609, -8.9322948, -5.7523098, -2.4629765, 2.4698505
6: -19.4389515, -15.5954771, -19.4166260, -15.5791779, -3.2746758, 3.2179499
7: 4.2413874, 6.9475818, 4.2751555, 6.9581480, -2.7167606, 2.6724262
8: -7.1590009, -4.4515944, -7.1396046, -4.4658523, -2.3892541, 2.3683701
9: -7.2057838, -3.7863615, -7.1867790, -3.8017640, -2.7442188, 2.7155495

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3074553, upper bound: 1.2980340
time: 5.41 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3074563, upper bound: 1.2992486
time: 5.47 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6000862, -13.5833101, -17.5848045, -13.5906773, -2.6760840, 2.6110599
1: -10.2804871, -7.4924779, -10.2407990, -7.5379009, -2.2670522, 2.2487173
2: -6.4222193, -3.5598989, -6.3695879, -3.6265078, -2.3039136, 2.3653667
3: -2.4288955, 0.1314921, -2.4066978, 0.1139760, -1.8325605, 1.8919678
4: -7.0419278, -2.9144959, -6.9735951, -2.9521682, -3.2196331, 3.1873708
5: -8.9853077, -5.7418566, -8.9478683, -5.7516832, -2.4742756, 2.4993086
6: -19.4483109, -15.5495968, -19.4181862, -15.5658455, -3.3018503, 3.2444801
7: 4.2326841, 6.9844837, 4.2735119, 6.9688697, -2.7361856, 2.7109718
8: -7.1711526, -4.4227982, -7.1420622, -4.4574766, -2.4112720, 2.3801038
9: -7.2127447, -3.7720175, -7.1884146, -3.7976503, -2.7565155, 2.7268329

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3104958, upper bound: 1.3064280
time: 5.26 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3104949, upper bound: 1.3076393
time: 5.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5951500, -13.5864258, -2.6848278, 2.6304853
1: -10.2822790, -7.4614220, -10.2611780, -7.4682827, -2.2902546, 2.2971687
2: -6.4601727, -3.5581918, -6.4552741, -3.6010275, -2.3836961, 2.3763762
3: -2.4422760, 0.1332504, -2.4361451, 0.1203668, -1.8933101, 1.9056354
4: -7.0440593, -2.8905525, -6.9880157, -2.8980706, -3.2374711, 3.2332711
5: -8.9876595, -5.7355232, -8.9580612, -5.7373915, -2.5205383, 2.5175791
6: -19.4601669, -15.5480900, -19.4452171, -15.5538960, -3.3323298, 3.2819481
7: 4.2270679, 6.9874468, 4.2631273, 6.9818163, -2.7547483, 2.7243195
8: -7.1751165, -4.3977709, -7.1660714, -4.4038391, -2.4239345, 2.4563551
9: -7.2168632, -3.7630367, -7.2067661, -3.7801449, -2.7823720, 2.7635641

Time for backsubstitution: 14.67 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7229785919189453
rel_dist={7: [-1.3105179015933643, 1.3105156598409975]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1841506, upper bound: 1.1791578
time: 7.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847103, upper bound: 1.1847087
time: 4.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.36 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.36
Output dim: 7, lower bound: -1.1841506, upper bound: 1.1791578
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.36
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

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791545
time: 4.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791550
time: 4.95 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1816144, upper bound: 1.1791819
time: 5.29 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847091, upper bound: 1.1847072
time: 4.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.95 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 24.95
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791545
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 24.95
Output dim: 7, lower bound: -1.1791557, upper bound: 1.1791550
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 24.95
Output dim: 7, lower bound: -1.1816144, upper bound: 1.1791819
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.95
Output dim: 7, lower bound: -1.1847091, upper bound: 1.1847072

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805225, -17.5972481, -13.5857992, -2.5366154, 2.4823139
1: -10.2822800, -7.4614229, -10.2654266, -7.4666901, -2.2746005, 2.2344279
2: -6.4601731, -3.5581913, -6.4559088, -3.5972743, -2.3207350, 2.3862247
3: -2.4422765, 0.1332507, -2.4377642, 0.1256822, -1.8223114, 1.8501265
4: -7.0440588, -2.8905525, -6.9938712, -2.8966632, -3.1829047, 3.1401196
5: -8.9876595, -5.7355232, -8.9602051, -5.7368994, -2.4454942, 2.4334912
6: -19.4601688, -15.5480900, -19.4462528, -15.5525637, -3.2123289, 3.1675344
7: 4.2270660, 6.9874468, 4.2598333, 6.9827938, -2.7557278, 2.7276134
8: -7.1751156, -4.3977699, -7.1687727, -4.4007759, -2.3974895, 2.3876011
9: -7.2168641, -3.7630351, -7.2100449, -3.7771645, -2.6891060, 2.6797814

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847080, upper bound: 1.1825599
time: 5.84 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847080, upper bound: 1.1847060
time: 5.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.63 seconds
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.63
Output dim: 7, lower bound: -1.1847080, upper bound: 1.1825599
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.63
Output dim: 7, lower bound: -1.1847080, upper bound: 1.1847060

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -17.5992355, -13.5838604, -17.5848045, -13.5906773, -2.5259871, 2.4660676
1: -10.2801294, -7.4985847, -10.2407980, -7.5379028, -2.2014389, 2.1728425
2: -6.4147630, -3.5602396, -6.3695855, -3.6265082, -2.2385669, 2.2976522
3: -2.4262733, 0.1311421, -2.4066992, 0.1139743, -1.7944088, 1.8172207
4: -7.0415039, -2.9192061, -6.9735942, -2.9521711, -3.1246824, 3.0910091
5: -8.9848423, -5.7430944, -8.9478664, -5.7516851, -2.4277411, 2.4110460
6: -19.4459801, -15.5498991, -19.4181862, -15.5658503, -3.1758566, 3.1383600
7: 4.2337818, 6.9838896, 4.2735138, 6.9688673, -2.7350855, 2.7103758
8: -7.1703582, -4.4277225, -7.1420593, -4.4574776, -2.3355885, 2.3186109
9: -7.2119193, -3.7737806, -7.1884108, -3.7976489, -2.6622586, 2.6369815

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809647, upper bound: 1.1748086
time: 6.36 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847058, upper bound: 1.1690149
time: 10.66 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805244, -17.5972443, -13.5857973, -2.5366139, 2.4810762
1: -10.2822790, -7.4614220, -10.2654276, -7.4666929, -2.2241406, 2.2344260
2: -6.4601727, -3.5581930, -6.4559059, -3.5972750, -2.3135214, 2.3020868
3: -2.4422748, 0.1332511, -2.4377630, 0.1256824, -1.8223095, 1.8318911
4: -7.0440574, -2.8905535, -6.9938736, -2.8966689, -3.1398273, 3.1401181
5: -8.9876595, -5.7355251, -8.9602032, -5.7369003, -2.4330168, 2.4334898
6: -19.4601631, -15.5480881, -19.4462471, -15.5525618, -3.2092590, 3.1553884
7: 4.2270670, 6.9874458, 4.2598338, 6.9827933, -2.7557263, 2.7276120
8: -7.1751137, -4.3977728, -7.1687708, -4.4007797, -2.3481812, 2.3840559
9: -7.2168646, -3.7630358, -7.2100439, -3.7771676, -2.6916752, 2.6739750

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847009, upper bound: 1.1833958
time: 5.12 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847007, upper bound: 1.1846986
time: 4.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.86 seconds
IS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 24.86
Output dim: 7, lower bound: -1.1809647, upper bound: 1.1748086
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 7, lower bound: -1.1847058, upper bound: 1.1690149
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 7, lower bound: -1.1847009, upper bound: 1.1833958
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 7, lower bound: -1.1847007, upper bound: 1.1846986

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.5992355, -13.5838594, -17.5848045, -13.5906773, -2.5237026, 2.4563348
1: -10.2801294, -7.4985862, -10.2407980, -7.5379028, -2.2014380, 2.1754212
2: -6.4147587, -3.5602379, -6.3695855, -3.6265082, -2.2245016, 2.2946665
3: -2.4262674, 0.1311431, -2.4066992, 0.1139743, -1.7519031, 1.8172204
4: -7.0415030, -2.9192059, -6.9735942, -2.9521711, -3.1230001, 3.0833540
5: -8.9848404, -5.7430954, -8.9478664, -5.7516851, -2.3836842, 2.4110456
6: -19.4459801, -15.5499010, -19.4181862, -15.5658503, -3.1758566, 3.1167507
7: 4.2337837, 6.9838877, 4.2735138, 6.9688673, -2.7350836, 2.7103739
8: -7.1703563, -4.4277229, -7.1420593, -4.4574776, -2.3352365, 2.2974801
9: -7.2119193, -3.7737811, -7.1884108, -3.7976489, -2.6622581, 2.6316972

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846987, upper bound: 1.1812819
time: 4.82 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1825500
time: 5.24 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805244, -17.5951481, -13.5864296, -2.5338945, 2.4773231
1: -10.2822790, -7.4614220, -10.2611771, -7.4682856, -2.2225714, 2.2300854
2: -6.4601727, -3.5581930, -6.4552736, -3.6010277, -2.3097367, 2.3013706
3: -2.4422748, 0.1332511, -2.4361451, 0.1203665, -1.8172898, 1.8304110
4: -7.0440574, -2.8905535, -6.9880133, -2.8980765, -3.1383896, 3.1343513
5: -8.9876595, -5.7355251, -8.9580593, -5.7373943, -2.4317379, 2.4308529
6: -19.4601631, -15.5480881, -19.4452171, -15.5538998, -3.2070589, 3.1535521
7: 4.2270670, 6.9874458, 4.2631264, 6.9818134, -2.7547464, 2.7243195
8: -7.1751137, -4.3977728, -7.1660686, -4.4038382, -2.3450861, 2.3813026
9: -7.2168646, -3.7630358, -7.2067642, -3.7801461, -2.6864586, 2.6688843

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809554, upper bound: 1.1756725
time: 5.94 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846987, upper bound: 1.1833934
time: 4.92 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -17.6044159, -13.5805254, -17.6048145, -13.5801544, -2.5413537, 2.5046306
1: -10.2822580, -7.4614277, -10.2709179, -7.4142847, -2.2411795, 2.2407737
2: -6.4601693, -3.5582018, -6.4833498, -3.5910392, -2.3198791, 2.3084702
3: -2.4422700, 0.1332269, -2.4767208, 0.1330500, -1.8321333, 1.8438592
4: -7.0440273, -2.8905578, -7.0033503, -2.8442817, -3.1546364, 3.1496401
5: -8.9876499, -5.7355223, -8.9705772, -5.7332106, -2.4401965, 2.4446115
6: -19.4601612, -15.5480957, -19.4527416, -15.5467005, -3.2225199, 3.1612272
7: 4.2270808, 6.9874420, 4.2460837, 7.0008883, -2.7738075, 2.7413583
8: -7.1751070, -4.3977842, -7.2139788, -4.3977222, -2.3520217, 2.4030054
9: -7.2168503, -3.7630444, -7.2608633, -3.7744663, -2.6923094, 2.7161117

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809552, upper bound: 1.1769449
time: 7.80 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1846956
time: 5.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.07 seconds
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 7, lower bound: -1.1846987, upper bound: 1.1812819
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1825500
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 7, lower bound: -1.1809554, upper bound: 1.1756725
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 7, lower bound: -1.1846987, upper bound: 1.1833934
IS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.07
Output dim: 7, lower bound: -1.1809552, upper bound: 1.1769449
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.07
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1846956

## BFS IS instance: IS_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.5992355, -13.5838594, -17.5826836, -13.5913181, -2.5209732, 2.4525759
1: -10.2801294, -7.4985862, -10.2365513, -7.5395060, -2.1998568, 2.1710792
2: -6.4147587, -3.5602379, -6.3689461, -3.6302605, -2.2207160, 2.2939534
3: -2.4262674, 0.1311431, -2.4050670, 0.1086578, -1.7468786, 1.8157644
4: -7.0415030, -2.9192059, -6.9677362, -2.9535916, -3.1215572, 3.0775871
5: -8.9848404, -5.7430954, -8.9457340, -5.7521791, -2.3824043, 2.4084272
6: -19.4459801, -15.5499010, -19.4171543, -15.5671740, -3.1736774, 3.1149139
7: 4.2337837, 6.9838877, 4.2768211, 6.9679003, -2.7341166, 2.7070665
8: -7.1703563, -4.4277229, -7.1393685, -4.4605374, -2.3321414, 2.2947216
9: -7.2119193, -3.7737811, -7.1851988, -3.8006272, -2.6570425, 2.6266899

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833957, upper bound: 1.1812822
time: 5.49 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833957, upper bound: 1.1812816
time: 5.15 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.5992279, -13.5838623, -17.5922756, -13.5850067, -2.5246847, 2.4807079
1: -10.2801075, -7.4985914, -10.2463388, -7.4856548, -2.2215290, 2.1818042
2: -6.4147563, -3.5602493, -6.3969307, -3.6202109, -2.2308989, 2.3010499
3: -2.4262607, 0.1311193, -2.4456925, 0.1214030, -1.7617154, 1.8252487
4: -7.0414729, -2.9192116, -6.9830818, -2.8998792, -3.1378121, 3.0928855
5: -8.9848328, -5.7430964, -8.9583788, -5.7479782, -2.3918362, 2.4222665
6: -19.4459763, -15.5499077, -19.4246826, -15.5598192, -3.1899071, 3.1225805
7: 4.2337952, 6.9838834, 4.2598143, 6.9871349, -2.7533398, 2.7240691
8: -7.1703482, -4.4277349, -7.1874390, -4.4544187, -2.3390856, 2.3235967
9: -7.2119069, -3.7737908, -7.2395601, -3.7949538, -2.6628985, 2.6882377

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1819895
time: 5.86 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1825507
time: 5.17 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6044273, -13.5805244, -17.5951481, -13.5864296, -2.5331106, 2.4675865
1: -10.2822790, -7.4614239, -10.2611771, -7.4682856, -2.2225695, 2.2326651
2: -6.4601674, -3.5581915, -6.4552736, -3.6010277, -2.2956715, 2.2983830
3: -2.4422705, 0.1332501, -2.4361451, 0.1203665, -1.7747827, 1.8304100
4: -7.0440559, -2.8905544, -6.9880133, -2.8980765, -3.1367092, 3.1266947
5: -8.9876537, -5.7355242, -8.9580593, -5.7373943, -2.3876810, 2.4308529
6: -19.4601650, -15.5480900, -19.4452171, -15.5538998, -3.2070589, 3.1319404
7: 4.2270679, 6.9874449, 4.2631264, 6.9818134, -2.7547455, 2.7243185
8: -7.1751142, -4.3977742, -7.1660686, -4.4038382, -2.3447332, 2.3601780
9: -7.2168651, -3.7630363, -7.2067642, -3.7801461, -2.6864567, 2.6635995

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833977, upper bound: 1.1833928
time: 4.89 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833977, upper bound: 1.1833929
time: 5.47 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6044159, -13.5805283, -17.6048145, -13.5801544, -2.5390692, 2.4945025
1: -10.2822561, -7.4614301, -10.2709179, -7.4142847, -2.2411785, 2.2433510
2: -6.4601669, -3.5582008, -6.4833498, -3.5910392, -2.3058133, 2.3054829
3: -2.4422653, 0.1332276, -2.4767208, 0.1330500, -1.7896271, 1.8389800
4: -7.0440273, -2.8905613, -7.0033503, -2.8442817, -3.1529541, 3.1419840
5: -8.9876480, -5.7355223, -8.9705772, -5.7332106, -2.3961005, 2.4446120
6: -19.4601593, -15.5480976, -19.4527416, -15.5467005, -3.2191534, 3.1396160
7: 4.2270789, 6.9874392, 4.2460837, 7.0008883, -2.7738094, 2.7413554
8: -7.1751060, -4.3977857, -7.2139788, -4.3977222, -2.3516693, 2.3816319
9: -7.2168493, -3.7630463, -7.2608633, -3.7744663, -2.6923094, 2.7108190

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1841355
time: 5.19 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1846964
time: 5.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 26.09 seconds
IS_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1833957, upper bound: 1.1812822
IS_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1833957, upper bound: 1.1812816
IS_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1819895
IS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1825507
IS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1833977, upper bound: 1.1833928
IS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1833977, upper bound: 1.1833929
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1841355
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 26.09
Output dim: 7, lower bound: -1.1791433, upper bound: 1.1846964

## BFS IS instance: IS_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.5971317, -13.5844946, -17.5826836, -13.5913181, -2.5172205, 2.4498594
1: -10.2758789, -7.5001845, -10.2365513, -7.5395060, -2.1955118, 2.1695037
2: -6.4141288, -3.5639963, -6.3689461, -3.6302605, -2.2199936, 2.2901640
3: -2.4246597, 0.1258256, -2.4050670, 0.1086578, -1.7454529, 1.8107395
4: -7.0356231, -2.9206386, -6.9677362, -2.9535916, -3.1157894, 3.0761452
5: -8.9826698, -5.7435923, -8.9457340, -5.7521791, -2.3797836, 2.4071431
6: -19.4449501, -15.5512409, -19.4171543, -15.5671740, -3.1718378, 3.1127262
7: 4.2370529, 6.9829149, 4.2768211, 6.9679003, -2.7308474, 2.7060938
8: -7.1676626, -4.4307847, -7.1393685, -4.4605374, -2.3293886, 2.2916279
9: -7.2086740, -3.7767673, -7.1851988, -3.8006272, -2.6520057, 2.6214662

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778392, upper bound: 1.1807229
time: 5.54 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778392, upper bound: 1.1812855
time: 8.41 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.6068001, -13.5781984, -17.5826836, -13.5913181, -2.5280399, 2.4613693
1: -10.2855730, -7.4476962, -10.2365513, -7.5395060, -2.2060480, 2.1901188
2: -6.4417667, -3.5541296, -6.3689461, -3.6302605, -2.2268467, 2.3001056
3: -2.4651160, 0.1384745, -2.4050670, 0.1086578, -1.7532279, 1.8249269
4: -7.0508585, -2.8681083, -6.9677362, -2.9535916, -3.1311102, 3.0955763
5: -8.9949837, -5.7394114, -8.9457340, -5.7521791, -2.3905697, 2.4160705
6: -19.4523239, -15.5440979, -19.4171543, -15.5671740, -3.1792173, 3.1197186
7: 4.2207060, 7.0019016, 4.2768211, 6.9679003, -2.7471943, 2.7250805
8: -7.2138476, -4.4246387, -7.1393685, -4.4605374, -2.3594275, 2.2984095
9: -7.2603521, -3.7711415, -7.1851988, -3.8006272, -2.7086921, 2.6272874

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1814978, upper bound: 1.1812819
time: 5.56 seconds

## Relational analysis of IS_A2_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1814978, upper bound: 1.1812820
time: 5.59 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -17.5992279, -13.5838623, -17.5995598, -13.5797138, -2.5280893, 2.4892306
1: -10.2801075, -7.4985914, -10.2631798, -7.4804096, -2.2246408, 2.1891193
2: -6.4147563, -3.5602493, -6.4011984, -3.5811229, -2.2374315, 2.3018284
3: -2.4262607, 0.1311193, -2.4503555, 0.1290569, -1.7690115, 1.8292465
4: -7.0414729, -2.9192116, -7.0332489, -2.8938379, -3.1356030, 3.1050758
5: -8.9848328, -5.7430964, -8.9857969, -5.7466059, -2.3932476, 2.4458520
6: -19.4459763, -15.5499077, -19.4386063, -15.5553827, -3.1954308, 3.1441422
7: 4.2337952, 6.9838834, 4.2271452, 6.9917583, -2.7579632, 2.7567382
8: -7.1703482, -4.4277349, -7.1937056, -4.4513888, -2.3425226, 2.3315434
9: -7.2119069, -3.7737908, -7.2462711, -3.7808037, -2.6820393, 2.6944816

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1778372, upper bound: 1.1825536
time: 11.09 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778372, upper bound: 1.1816315
time: 5.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -17.6023254, -13.5811510, -17.5951481, -13.5864296, -2.5293646, 2.4648709
1: -10.2780266, -7.4630208, -10.2611771, -7.4682856, -2.2182255, 2.2310772
2: -6.4595399, -3.5619502, -6.4552736, -3.6010277, -2.2949572, 2.2945943
3: -2.4406669, 0.1279333, -2.4361451, 0.1203665, -1.7733588, 1.8253865
4: -7.0381804, -2.8919849, -6.9880133, -2.8980765, -3.1309414, 3.1252546
5: -8.9854870, -5.7360163, -8.9580593, -5.7373943, -2.3850527, 2.4295683
6: -19.4591331, -15.5494347, -19.4452171, -15.5538998, -3.2052202, 3.1297669
7: 4.2303238, 6.9864659, 4.2631264, 6.9818134, -2.7514896, 2.7233396
8: -7.1724129, -4.4008341, -7.1660686, -4.4038382, -2.3420010, 2.3570838
9: -7.2135839, -3.7660227, -7.2067642, -3.7801461, -2.6814604, 2.6583753

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1821631, upper bound: 1.1798204
time: 5.20 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1820754, upper bound: 1.1820711
time: 4.90 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -17.6120110, -13.5748749, -17.5951481, -13.5864296, -2.5402765, 2.4763584
1: -10.2877007, -7.4104409, -10.2611771, -7.4682856, -2.2287474, 2.2503510
2: -6.4872322, -3.5521150, -6.4552736, -3.6010277, -2.3018236, 2.3045125
3: -2.4811058, 0.1405708, -2.4361451, 0.1203665, -1.7820997, 1.8395386
4: -7.0534067, -2.8393965, -6.9880133, -2.8980765, -3.1462574, 3.1436553
5: -8.9977360, -5.7318439, -8.9580593, -5.7373943, -2.3957624, 2.4384871
6: -19.4665031, -15.5423851, -19.4452171, -15.5538998, -3.2125940, 3.1368327
7: 4.2140255, 7.0053906, 4.2631264, 6.9818134, -2.7677879, 2.7422643
8: -7.2185564, -4.3946919, -7.1660686, -4.4038382, -2.3714423, 2.3638611
9: -7.2652712, -3.7603974, -7.2067642, -3.7801461, -2.7330427, 2.6642022

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1821631, upper bound: 1.1798202
time: 5.68 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1820754, upper bound: 1.1820709
time: 5.26 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6042557, -13.5809574, -17.5957756, -13.5844393, -2.5302982, 2.4848168
1: -10.2821522, -7.4614463, -10.2678938, -7.4243231, -2.2306547, 2.2395716
2: -6.4601259, -3.5582814, -6.4652944, -3.5933995, -2.2990255, 2.2871721
3: -2.4422641, 0.1331642, -2.4730272, 0.1256086, -1.7763171, 1.8323801
4: -7.0437055, -2.8905969, -6.9978662, -2.8662114, -3.1301246, 3.1367297
5: -8.9874773, -5.7355251, -8.9641905, -5.7420702, -2.3857827, 2.4381809
6: -19.4600639, -15.5481243, -19.4492702, -15.5561695, -3.2025290, 3.1345568
7: 4.2272892, 6.9873972, 4.2508211, 6.9848084, -2.7575192, 2.7365761
8: -7.1750069, -4.3978596, -7.2070017, -4.3999300, -2.3443389, 2.3727729
9: -7.2168016, -3.7636111, -7.2524767, -3.7756188, -2.6886806, 2.6974163

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778845, upper bound: 1.1805250
time: 5.89 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1777917, upper bound: 1.1827934
time: 4.98 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6044159, -13.5805283, -17.6120968, -13.5748587, -2.5424824, 2.5030303
1: -10.2822561, -7.4614301, -10.2877502, -7.4090471, -2.2443404, 2.2498450
2: -6.4601669, -3.5582008, -6.4876065, -3.5519624, -2.3123636, 2.3062644
3: -2.4422653, 0.1332276, -2.4813504, 0.1407342, -1.7969294, 1.8429964
4: -7.0440273, -2.8905613, -7.0535259, -2.8382273, -3.1507692, 3.1541252
5: -8.9876480, -5.7355223, -8.9979925, -5.7318306, -2.3975210, 2.4658022
6: -19.4601593, -15.5480976, -19.4666367, -15.5422554, -3.2246680, 3.1611385
7: 4.2270789, 6.9874392, 4.2134018, 7.0055590, -2.7784801, 2.7740374
8: -7.1751060, -4.3977857, -7.2203140, -4.3946891, -2.3551230, 2.3896220
9: -7.2168493, -3.7630463, -7.2676535, -3.7603204, -2.7114553, 2.7171025

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778845, upper bound: 1.1811487
time: 5.20 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1777917, upper bound: 1.1834123
time: 5.38 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 25.31 seconds
IS_A2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1778392, upper bound: 1.1807229
IS_A2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1778392, upper bound: 1.1812855
IS_A2_A2_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1814978, upper bound: 1.1812819
IS_A2_A2_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1814978, upper bound: 1.1812820
IS_A2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1778372, upper bound: 1.1825536
IS_A2_A2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1778372, upper bound: 1.1816315
IS_A2_A2_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1821631, upper bound: 1.1798204
IS_A2_A2_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1820754, upper bound: 1.1820711
IS_A2_A2_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1821631, upper bound: 1.1798202
IS_A2_A2_B2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1820754, upper bound: 1.1820709
IS_A2_A2_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1778845, upper bound: 1.1805250
IS_A2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1777917, upper bound: 1.1827934
IS_A2_A2_B2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1778845, upper bound: 1.1811487
IS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 25.31
Output dim: 7, lower bound: -1.1777917, upper bound: 1.1834123

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -17.5971317, -13.5844946, -17.5994759, -13.5797138, -2.5240958, 2.4691725
1: -10.2758789, -7.5001845, -10.2631302, -7.4817944, -2.2192936, 2.1873913
2: -6.4141288, -3.5639963, -6.4008307, -3.5812769, -2.2364354, 2.2978137
3: -2.4246597, 0.1258256, -2.4501190, 0.1288941, -1.7668581, 1.8235717
4: -7.0356231, -2.9206386, -7.0331306, -2.8950038, -3.1289301, 3.1033115
5: -8.9826698, -5.7435923, -8.9855442, -5.7466192, -2.3901248, 2.4439840
6: -19.4449501, -15.5512409, -19.4384766, -15.5555115, -3.1843796, 3.1416254
7: 4.2370529, 6.9829149, 4.2277660, 6.9915895, -2.7545366, 2.7551489
8: -7.1676626, -4.4307847, -7.1919470, -4.4513884, -2.3395777, 2.3272493
9: -7.2086740, -3.7767673, -7.2438784, -3.7808752, -2.6769500, 2.6846404

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1770183, upper bound: 1.1825512
time: 5.27 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1770183, upper bound: 1.1825507
time: 5.26 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -17.6042480, -13.5841999, -17.5936985, -13.5898685, -2.5258756, 2.4833508
1: -10.2821503, -7.4614744, -10.2729712, -7.4241991, -2.2280722, 2.2398441
2: -6.4601064, -3.5582876, -6.4653950, -3.5913200, -2.2992039, 2.2861536
3: -2.4422610, 0.1329271, -2.4730220, 0.1256496, -1.7763047, 1.8321428
4: -7.0437036, -2.8906450, -7.0037851, -2.8660262, -3.1265740, 3.1367378
5: -8.9868832, -5.7355270, -8.9636431, -5.7431021, -2.3857756, 2.4382396
6: -19.4599915, -15.5481300, -19.4491711, -15.5551090, -3.2035275, 3.1350565
7: 4.2273092, 6.9873886, 4.2508597, 6.9870758, -2.7597666, 2.7365289
8: -7.1749864, -4.3978906, -7.2109795, -4.3999200, -2.3462696, 2.3719647
9: -7.2167978, -3.7636547, -7.2584143, -3.7756975, -2.6922770, 2.6960578

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1764523, upper bound: 1.1827934
time: 5.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1764523, upper bound: 1.1820306
time: 5.99 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -17.6044083, -13.5837688, -17.6100578, -13.5802889, -2.5380609, 2.5008819
1: -10.2822533, -7.4614577, -10.2928314, -7.4089236, -2.2417562, 2.2501171
2: -6.4601469, -3.5582080, -6.4877071, -3.5498824, -2.3125410, 2.3052456
3: -2.4422617, 0.1329916, -2.4813433, 0.1407700, -1.7969146, 1.8427598
4: -7.0440254, -2.8906047, -7.0594435, -2.8380418, -3.1472163, 3.1542497
5: -8.9870529, -5.7355242, -8.9974413, -5.7328634, -2.3975134, 2.4658790
6: -19.4600906, -15.5480986, -19.4665337, -15.5411987, -3.2247362, 3.1616354
7: 4.2271004, 6.9874325, 4.2134428, 7.0078249, -2.7807245, 2.7739897
8: -7.1750855, -4.3978171, -7.2242880, -4.3946772, -2.3570542, 2.3888147
9: -7.2168474, -3.7630892, -7.2735896, -3.7603962, -2.7150507, 2.7157409

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1775797, upper bound: 1.1834124
time: 4.92 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1775797, upper bound: 1.1820715
time: 5.94 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 25.82 seconds
IS_A2_A2_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 25.82
Output dim: 7, lower bound: -1.1770183, upper bound: 1.1825512
IS_A2_A2_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 25.82
Output dim: 7, lower bound: -1.1770183, upper bound: 1.1825507
IS_A2_A2_B2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 25.82
Output dim: 7, lower bound: -1.1764523, upper bound: 1.1827934
IS_A2_A2_B2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 25.82
Output dim: 7, lower bound: -1.1764523, upper bound: 1.1820306
IS_A2_A2_B2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 25.82
Output dim: 7, lower bound: -1.1775797, upper bound: 1.1834124
IS_A2_A2_B2_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 25.82
Output dim: 7, lower bound: -1.1775797, upper bound: 1.1820715

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -17.5898666, -13.5860424, -17.5994759, -13.5797138, -2.5186179, 2.4680886
1: -10.2534084, -7.5342340, -10.2631302, -7.4817944, -2.1969042, 2.1534147
2: -6.3732238, -3.5911701, -6.4008307, -3.5812769, -2.2200503, 2.2706513
3: -2.4096231, 0.1161985, -2.4501190, 0.1288941, -1.7523923, 1.8132515
4: -7.0178890, -2.9475160, -7.0331306, -2.8950038, -3.1111012, 3.0763903
5: -8.9731655, -5.7508078, -8.9855442, -5.7466192, -2.3792000, 2.4401863
6: -19.4310951, -15.5627232, -19.4384766, -15.5555115, -3.1771030, 3.1269979
7: 4.2440329, 6.9725075, 4.2277660, 6.9915895, -2.7475567, 2.7447414
8: -7.1456423, -4.4575405, -7.1919470, -4.4513884, -2.3143597, 2.3170059
9: -7.1919827, -3.7865028, -7.2438784, -3.7808752, -2.6553664, 2.6726274

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1692774, upper bound: 1.1788094
time: 6.51 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1692771, upper bound: 1.1825543
time: 6.17 seconds

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -17.6023235, -13.5811539, -17.5994759, -13.5797138, -2.5264480, 2.4727974
1: -10.2780247, -7.4630218, -10.2631302, -7.4817944, -2.2014875, 2.1925015
2: -6.4595380, -3.5619507, -6.4008307, -3.5812769, -2.2393093, 2.2752426
3: -2.4406655, 0.1279321, -2.4501190, 0.1288941, -1.7704399, 1.8191757
4: -7.0381780, -2.8919902, -7.0331306, -2.8950038, -3.1162729, 3.1082296
5: -8.9854832, -5.7360191, -8.9855442, -5.7466192, -2.3879857, 2.4455771
6: -19.4591274, -15.5494347, -19.4384766, -15.5555115, -3.1971464, 3.1408987
7: 4.2303257, 6.9864626, 4.2277660, 6.9915895, -2.7612638, 2.7586966
8: -7.1724119, -4.4008350, -7.1919470, -4.4513884, -2.3398190, 2.3280625
9: -7.2135839, -3.7660251, -7.2438784, -3.7808752, -2.6779895, 2.6785603

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 73

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1692771, upper bound: 1.1788071
time: 6.84 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1692790, upper bound: 1.1825536
time: 6.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -17.6021595, -13.5848255, -17.5936241, -13.5898666, -2.5218859, 2.4621685
1: -10.2779217, -7.4630666, -10.2729225, -7.4255886, -2.2227218, 2.2381229
2: -6.4594784, -3.5620353, -6.4650240, -3.5914741, -2.2982178, 2.2821407
3: -2.4406614, 0.1276321, -2.4727864, 0.1254890, -1.7740936, 1.8264639
4: -7.0378537, -2.8920689, -7.0036645, -2.8671999, -3.1198940, 3.1349812
5: -8.9847212, -5.7360210, -8.9633865, -5.7431164, -2.3828096, 2.4340072
6: -19.4589691, -15.5494671, -19.4490337, -15.5552349, -3.1923628, 3.1325579
7: 4.2305551, 6.9864159, 4.2514772, 6.9869137, -2.7563586, 2.7349386
8: -7.1722956, -4.4009399, -7.2092257, -4.3999214, -2.3433428, 2.3676755
9: -7.2135344, -3.7666292, -7.2560310, -3.7757704, -2.6872277, 2.6864376

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1686765, upper bound: 1.1790170
time: 5.06 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1686765, upper bound: 1.1827938
time: 5.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -17.6023197, -13.5843964, -17.6099739, -13.5802889, -2.5340714, 2.4798059
1: -10.2780256, -7.4630499, -10.2927818, -7.4103112, -2.2364063, 2.2483950
2: -6.4595189, -3.5619564, -6.4873362, -3.5500340, -2.3115530, 2.3012314
3: -2.4406641, 0.1276950, -2.4811068, 0.1406089, -1.7947025, 1.8370852
4: -7.0381746, -2.8920331, -7.0593266, -2.8392086, -3.1405411, 3.1524854
5: -8.9848919, -5.7360172, -8.9971943, -5.7328777, -2.3945475, 2.4640038
6: -19.4590626, -15.5494375, -19.4664001, -15.5413198, -3.2186537, 3.1591325
7: 4.2303462, 6.9864569, 4.2140656, 7.0076613, -2.7773151, 2.7723913
8: -7.1723928, -4.4008646, -7.2225332, -4.3946786, -2.3541298, 2.3845260
9: -7.2135825, -3.7660637, -7.2712126, -3.7604704, -2.7100019, 2.7061181

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 539

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1698084, upper bound: 1.1796449
time: 6.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1698084, upper bound: 1.1834130
time: 4.91 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 26.28 seconds
IS_A2_A2_B1_A2_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1692774, upper bound: 1.1788094
IS_A2_A2_B1_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1692771, upper bound: 1.1825543
IS_A2_A2_B1_A2_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1692771, upper bound: 1.1788071
IS_A2_A2_B1_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1692790, upper bound: 1.1825536
IS_A2_A2_B2_B2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1686765, upper bound: 1.1790170
IS_A2_A2_B2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1686765, upper bound: 1.1827938
IS_A2_A2_B2_B2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1698084, upper bound: 1.1796449
IS_A2_A2_B2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 26.28
Output dim: 7, lower bound: -1.1698084, upper bound: 1.1834130

## BFS IS instance: IS_A2_A2_B1_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -17.5898666, -13.5860424, -17.5994778, -13.5797176, -2.5107026, 2.4680872
1: -10.2534084, -7.5342340, -10.2631302, -7.4817924, -2.1988728, 2.1534138
2: -6.3732238, -3.5911701, -6.4008269, -3.5812776, -2.2186632, 2.2580836
3: -2.4096231, 0.1161985, -2.4501166, 0.1288919, -1.7523928, 1.7752259
4: -7.0178890, -2.9475160, -7.0331302, -2.8950062, -3.1042552, 3.0763917
5: -8.9731655, -5.7508078, -8.9855423, -5.7466192, -2.3789506, 2.3999271
6: -19.4310951, -15.5627232, -19.4384766, -15.5555134, -3.1554928, 3.1269975
7: 4.2440329, 6.9725075, 4.2277670, 6.9915857, -2.7475529, 2.7447405
8: -7.1456423, -4.4575405, -7.1919465, -4.4513927, -2.2935147, 2.3170056
9: -7.1919827, -3.7865028, -7.2438784, -3.7808771, -2.6500821, 2.6723070

Time for backsubstitution: 14.46 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.1847181417998263, 1.1847155369154763]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2417.89 seconds
