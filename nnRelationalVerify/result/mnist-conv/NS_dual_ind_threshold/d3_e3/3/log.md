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
execution time: IAR + RelationalAnalysis = 23.08 + 34.10 = 57.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.4415111, upper bound: 0.4415114

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4243539
time: 3.17 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697
time: 2.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.12 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.12
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4243539
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.12
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.8448048, -10.9546204, -12.8452301, -10.9188919, -1.0825844, 1.0478451
1: -10.8968830, -8.8450069, -10.8973999, -8.8077011, -1.2993164, 1.2713077
2: -10.4901161, -8.9427214, -10.4901285, -8.9274025, -1.2114487, 1.1955216
3: -4.1868286, -2.7238634, -4.1869063, -2.6790941, -0.8986335, 0.8702087
4: -14.8858490, -12.9758263, -14.9115801, -12.9757957, -1.0774136, 1.0927110
5: 8.5349770, 9.5771732, 8.5341835, 9.5872631, -0.6945202, 0.6787934
6: -4.3714967, -2.6373656, -4.3722811, -2.6192470, -1.0343199, 1.0139532
7: -15.4906120, -13.4622774, -15.4907885, -13.4439487, -1.2948971, 1.2719221
8: -0.5478458, 0.6971941, -0.5649807, 0.6974404, -0.7250931, 0.7500842
9: -6.4696803, -5.2913551, -6.4851003, -5.2912593, -0.7647269, 0.7726619

Time for backsubstitution: 8.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697
time: 2.96 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697
time: 2.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.9673119, -10.9315548, -12.8441839, -10.9242373, -1.2044182, 1.0879743
1: -11.0059433, -8.8669910, -10.8961535, -8.8313437, -1.4396911, 1.2936354
2: -10.5333252, -8.9543600, -10.4900980, -8.9379921, -1.2733543, 1.2004302
3: -4.3193283, -2.7348771, -4.1867194, -2.6976004, -1.0921144, 0.8953975
4: -14.8877945, -12.9076233, -14.9028988, -12.9758654, -1.0891514, 1.1685753
5: 8.5260372, 9.5431070, 8.5360317, 9.5723248, -0.7483356, 0.6719414
6: -4.4157004, -2.6483788, -4.3704834, -2.6297221, -1.1038425, 1.0190256
7: -15.5324774, -13.5037384, -15.4903688, -13.4664602, -1.3541536, 1.2778082
8: -0.5204105, 0.7454383, -0.5453670, 0.6968625, -0.7348642, 0.8567519
9: -6.4670277, -5.2453690, -6.4786797, -5.2914877, -0.7748582, 0.8245564

Time for backsubstitution: 8.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4154247, upper bound: 0.4099626
time: 2.90 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4154247, upper bound: 0.4154262
time: 2.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.99 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.99
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.99
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.99
Output dim: 5, lower bound: -0.4154247, upper bound: 0.4099626
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.99
Output dim: 5, lower bound: -0.4154247, upper bound: 0.4154262

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.8448048, -10.9546204, -12.8448048, -10.9546204, -1.0474157, 1.0474157
1: -10.8968830, -8.8450069, -10.8968830, -8.8450069, -1.2708611, 1.2708611
2: -10.4901161, -8.9427214, -10.4901161, -8.9427214, -1.1950686, 1.1950688
3: -4.1868286, -2.7238634, -4.1868286, -2.7238634, -0.8701663, 0.8701663
4: -14.8858490, -12.9758263, -14.8858490, -12.9758263, -1.0773950, 1.0773952
5: 8.5349770, 9.5771732, 8.5349770, 9.5771732, -0.6765118, 0.6765118
6: -4.3714967, -2.6373656, -4.3714967, -2.6373656, -1.0132775, 1.0132780
7: -15.4906120, -13.4622774, -15.4906120, -13.4622774, -1.2718887, 1.2718887
8: -0.5478458, 0.6971941, -0.5478458, 0.6971941, -0.7245221, 0.7245219
9: -6.4696803, -5.2913551, -6.4696803, -5.2913551, -0.7646706, 0.7646706

Time for backsubstitution: 8.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4099612, upper bound: 0.4202104
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4154246, upper bound: 0.4202103
time: 3.14 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.8448048, -10.9546204, -12.9673119, -10.9315548, -1.0678754, 1.1752634
1: -10.8968830, -8.8450069, -11.0059433, -8.8669910, -1.2848482, 1.4200325
2: -10.4901161, -8.9427214, -10.5333252, -8.9543600, -1.1974163, 1.2620277
3: -4.1868286, -2.7238634, -4.3193283, -2.7348771, -0.8917465, 1.0661683
4: -14.8858490, -12.9758263, -14.8877945, -12.9076233, -1.1557770, 1.0873547
5: 8.5349770, 9.5771732, 8.5260372, 9.5431070, -0.6940417, 0.7296236
6: -4.3714967, -2.6373656, -4.4157004, -2.6483788, -1.0197103, 1.0887871
7: -15.4906120, -13.4622774, -15.5324774, -13.5037384, -1.2673187, 1.3425441
8: -0.5478458, 0.6971941, -0.5204105, 0.7454383, -0.8310149, 0.7491786
9: -6.4696803, -5.2913551, -6.4670277, -5.2453690, -0.8204331, 0.7661371

Time for backsubstitution: 8.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4099612, upper bound: 0.4202104
time: 2.77 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4154246, upper bound: 0.4202104
time: 3.08 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.9673023, -10.9380894, -12.8505173, -10.9411230, -1.1797175, 1.0705993
1: -11.0059366, -8.8688850, -10.9167767, -8.8361607, -1.4353428, 1.3068304
2: -10.5333223, -8.9661980, -10.5137701, -8.9599991, -1.2377396, 1.1735406
3: -4.3082490, -2.7348843, -4.1617389, -2.6898725, -1.0700078, 0.8684504
4: -14.8877926, -12.9433451, -14.8921566, -13.0468044, -1.0238473, 1.1213670
5: 8.5422421, 9.5431080, 8.5629482, 9.5784416, -0.7262100, 0.6368624
6: -4.4053993, -2.6483819, -4.3455868, -2.6336966, -1.0872431, 0.9973068
7: -15.5324783, -13.5078526, -15.4987583, -13.4755993, -1.3448944, 1.2808561
8: -0.5077577, 0.7454369, -0.5123169, 0.6882339, -0.7127659, 0.8233161
9: -6.4670277, -5.2659445, -6.4814448, -5.3431449, -0.7091646, 0.7759769

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4015361
time: 4.02 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4046976
time: 2.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.9673080, -10.9322882, -12.8441706, -10.9292812, -1.1744375, 1.0877597
1: -11.0059423, -8.8679323, -10.8961344, -8.8391533, -1.4453497, 1.2926633
2: -10.5333233, -8.9557762, -10.4900875, -8.9497566, -1.2262516, 1.1994956
3: -4.3182621, -2.7348781, -4.1805811, -2.6976173, -1.0914929, 0.8670219
4: -14.8877926, -12.9080954, -14.9029007, -12.9794188, -1.0274227, 1.1684356
5: 8.5264854, 9.5431070, 8.5392475, 9.5723228, -0.7481401, 0.6399461
6: -4.4154882, -2.6483793, -4.3687253, -2.6297221, -1.1037061, 0.9969161
7: -15.5324821, -13.5042019, -15.4903660, -13.4703178, -1.3587451, 1.2772617
8: -0.5202231, 0.7454383, -0.5440228, 0.6968632, -0.7347519, 0.8224988
9: -6.4670277, -5.2461982, -6.4786797, -5.2983670, -0.6940117, 0.8245246

Time for backsubstitution: 8.57 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4070009
time: 2.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4101611
time: 2.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.34 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4099612, upper bound: 0.4202104
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4154246, upper bound: 0.4202103
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4099612, upper bound: 0.4202104
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4154246, upper bound: 0.4202104
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4015361
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4046976
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4070009
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.34
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4101611

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.8511276, -10.9715042, -12.8447943, -10.9611549, -1.0300264, 1.0227151
1: -10.9175053, -8.8498259, -10.8968754, -8.8469019, -1.2840552, 1.2665143
2: -10.5137920, -8.9647150, -10.4901142, -8.9545574, -1.1681960, 1.1594586
3: -4.1618514, -2.7161355, -4.1757498, -2.7238710, -0.8432226, 0.8480524
4: -14.8751011, -13.0467606, -14.8858461, -13.0115471, -1.0301762, 1.0120933
5: 8.5618849, 9.5832844, 8.5511417, 9.5771723, -0.6414467, 0.6544268
6: -4.3466144, -2.6413412, -4.3611989, -2.6373684, -0.9915752, 0.9966688
7: -15.4989929, -13.4714165, -15.4906111, -13.4663916, -1.2749209, 1.2626281
8: -0.5147955, 0.6885638, -0.5351930, 0.6971927, -0.6910787, 0.7024263
9: -6.4724445, -5.3430104, -6.4696803, -5.3119307, -0.7160921, 0.6989770

Time for backsubstitution: 8.50 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4138046, upper bound: 0.4215793
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4169647, upper bound: 0.4224298
time: 2.95 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.8447895, -10.9596624, -12.8448009, -10.9553547, -1.0472021, 1.0173805
1: -10.8968630, -8.8528147, -10.8968811, -8.8459492, -1.2698898, 1.2765107
2: -10.4901085, -8.9544888, -10.4901180, -8.9441395, -1.1941335, 1.1479733
3: -4.1806917, -2.7238798, -4.1857634, -2.7238648, -0.8417239, 0.8695436
4: -14.8858471, -12.9793758, -14.8858471, -12.9762936, -1.0772543, 1.0156026
5: 8.5381985, 9.5771713, 8.5354347, 9.5771723, -0.6444893, 0.6763166
6: -4.3697433, -2.6373658, -4.3712873, -2.6373668, -0.9911637, 1.0131397
7: -15.4906101, -13.4661350, -15.4906149, -13.4627428, -1.2713413, 1.2764208
8: -0.5465031, 0.6971927, -0.5476577, 0.6971936, -0.6902541, 0.7244095
9: -6.4696803, -5.2982335, -6.4696803, -5.2921848, -0.7646391, 0.6838236

Time for backsubstitution: 8.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4192679, upper bound: 0.4215809
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4224281, upper bound: 0.4224297
time: 2.95 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.8511276, -10.9715042, -12.9673023, -10.9380894, -1.0504861, 1.1505630
1: -10.9175053, -8.8498259, -11.0059366, -8.8688850, -1.2980409, 1.4156857
2: -10.5137920, -8.9647150, -10.5333223, -8.9661980, -1.1705246, 1.2264175
3: -4.1618514, -2.7161355, -4.3082490, -2.7348843, -0.8648033, 1.0440392
4: -14.8751011, -13.0467606, -14.8877926, -12.9433451, -1.1085529, 1.0220523
5: 8.5618849, 9.5832844, 8.5422421, 9.5431080, -0.6589766, 0.7074881
6: -4.3466144, -2.6413412, -4.4053993, -2.6483819, -0.9980085, 1.0721774
7: -15.4989929, -13.4714165, -15.5324783, -13.5078526, -1.2703543, 1.3332820
8: -0.5147955, 0.6885638, -0.5077577, 0.7454369, -0.7975717, 0.7270784
9: -6.4724445, -5.3430104, -6.4670277, -5.2659445, -0.7718537, 0.7004437

Time for backsubstitution: 8.61 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4015361, upper bound: 0.4140964
time: 3.06 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4046962, upper bound: 0.4149454
time: 3.11 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.8447895, -10.9596624, -12.9673080, -10.9322882, -1.0676618, 1.1452825
1: -10.8968630, -8.8528147, -11.0059423, -8.8679323, -1.2838764, 1.4256921
2: -10.4901085, -8.9544888, -10.5333233, -8.9557762, -1.1964803, 1.2149298
3: -4.1806917, -2.7238798, -4.3182621, -2.7348781, -0.8633742, 1.0655446
4: -14.8858471, -12.9793758, -14.8877926, -12.9080954, -1.1556373, 1.0256286
5: 8.5381985, 9.5771713, 8.5264854, 9.5431070, -0.6620605, 0.7294283
6: -4.3697433, -2.6373658, -4.4154882, -2.6483793, -0.9976182, 1.0886505
7: -15.4906101, -13.4661350, -15.5324821, -13.5042019, -1.2667718, 1.3471332
8: -0.5465031, 0.6971927, -0.5202231, 0.7454383, -0.7967584, 0.7490671
9: -6.4696803, -5.2982335, -6.4670277, -5.2461982, -0.8204005, 0.6852906

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4069994, upper bound: 0.4140949
time: 2.94 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4149439
time: 2.90 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.9637556, -10.9540405, -12.8491936, -10.9482164, -1.1654015, 1.0436835
1: -11.0010891, -8.8931217, -10.9150496, -8.8452234, -1.4141412, 1.2634296
2: -10.4862576, -8.9701271, -10.4937983, -8.9614983, -1.1810951, 1.1506181
3: -4.3066735, -2.7368939, -4.1611600, -2.6906517, -1.0682387, 0.8666437
4: -14.8876791, -12.9490643, -14.8921089, -13.0489178, -1.0188515, 1.1127861
5: 8.5427952, 9.5266609, 8.5631628, 9.5723820, -0.7148883, 0.6175772
6: -4.4040461, -2.6726327, -4.3450274, -2.6432495, -1.0742230, 0.9656775
7: -15.5284920, -13.5177937, -15.4975529, -13.4792805, -1.3389196, 1.2702405
8: -0.4893105, 0.7452798, -0.5055032, 0.6881728, -0.6922133, 0.8122389
9: -6.4604692, -5.2679105, -6.4790292, -5.3438592, -0.6970601, 0.7702024

Time for backsubstitution: 8.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4015367
time: 4.04 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4015361
time: 4.27 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.9850712, -10.9452896, -12.8499374, -10.9441299, -1.2127051, 1.0500748
1: -11.0082388, -8.9039669, -10.9160900, -8.8528519, -1.4814367, 1.2619143
2: -10.5175104, -8.9255390, -10.5052109, -8.9604607, -1.1949027, 1.2342367
3: -4.3115692, -2.7338181, -4.1615620, -2.6902719, -1.0716386, 0.8695600
4: -14.8852463, -12.9547081, -14.8921356, -13.0515499, -1.0239012, 1.1152864
5: 8.5334568, 9.5298319, 8.5630617, 9.5728683, -0.7298543, 0.6261983
6: -4.4296155, -2.6528783, -4.3452873, -2.6372142, -1.1207020, 0.9749479
7: -15.5361376, -13.5162754, -15.4983854, -13.4797220, -1.3464627, 1.2733488
8: -0.5036125, 0.7654667, -0.5097482, 0.6882064, -0.7088767, 0.8351874
9: -6.4591236, -5.2655802, -6.4779768, -5.3434181, -0.7005539, 0.7907307

Time for backsubstitution: 8.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4046974
time: 2.93 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4046976
time: 2.84 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.9637623, -10.9482374, -12.8428593, -10.9363766, -1.1601486, 1.0608313
1: -11.0010967, -8.8921719, -10.8944063, -8.8482151, -1.4241543, 1.2492614
2: -10.4862585, -8.9596939, -10.4701090, -8.9511881, -1.1695538, 1.1766391
3: -4.3166871, -2.7368884, -4.1800270, -2.6983888, -1.0897739, 0.8652056
4: -14.8876801, -12.9138145, -14.9028511, -12.9815340, -1.0224349, 1.1598616
5: 8.5270433, 9.5266628, 8.5394707, 9.5662422, -0.7367554, 0.6207974
6: -4.4141335, -2.6726313, -4.3681602, -2.6392760, -1.0906286, 0.9653451
7: -15.5284920, -13.5141506, -15.4890938, -13.4740028, -1.3529596, 1.2665663
8: -0.5017762, 0.7452807, -0.5371926, 0.6967988, -0.7141690, 0.8114276
9: -6.4604692, -5.2481637, -6.4762640, -5.2990780, -0.6819072, 0.8187494

Time for backsubstitution: 8.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4070009
time: 2.91 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4070009
time: 2.84 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.9850826, -10.9394875, -12.8436079, -10.9322901, -1.2075353, 1.0672328
1: -11.0082455, -8.9030142, -10.8954487, -8.8558435, -1.4913945, 1.2477508
2: -10.5175114, -8.9151115, -10.4815245, -8.9501953, -1.1833167, 1.2602284
3: -4.3215833, -2.7338140, -4.1804099, -2.6980104, -1.0931485, 0.8681186
4: -14.8852501, -12.9194565, -14.9028778, -12.9841661, -1.0274997, 1.1623483
5: 8.5177164, 9.5298319, 8.5393658, 9.5667439, -0.7517350, 0.6293495
6: -4.4397039, -2.6528778, -4.3684216, -2.6332390, -1.1371112, 0.9747343
7: -15.5361385, -13.5126438, -15.4899750, -13.4744453, -1.3604059, 1.2697043
8: -0.5160785, 0.7654674, -0.5414467, 0.6968327, -0.7308288, 0.8343861
9: -6.4591236, -5.2458324, -6.4752150, -5.2986403, -0.6854010, 0.8392787

Time for backsubstitution: 8.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4101611
time: 3.01 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4101611
time: 2.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.98 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4138046, upper bound: 0.4215793
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4169647, upper bound: 0.4224298
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4192679, upper bound: 0.4215809
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4224281, upper bound: 0.4224297
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4015361, upper bound: 0.4140964
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4046962, upper bound: 0.4149454
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4069994, upper bound: 0.4140949
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4149439
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4015367
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4015361
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4046974
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4046976
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4070009
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4093108, upper bound: 0.4070009
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4101611
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 5, lower bound: -0.4101598, upper bound: 0.4101611

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.8498096, -10.9785995, -12.8413706, -10.9771080, -1.0031137, 1.0085406
1: -10.9157867, -8.8588867, -10.8922968, -8.8711367, -1.2406516, 1.2458832
2: -10.4938192, -8.9662266, -10.4430647, -8.9584856, -1.1452994, 1.1027966
3: -4.1612740, -2.7169170, -4.1742306, -2.7258773, -0.8414128, 0.8465881
4: -14.8750553, -13.0488749, -14.8857327, -13.0172672, -1.0217514, 1.0070968
5: 8.5621004, 9.5772228, 8.5516882, 9.5607033, -0.6221479, 0.6429653
6: -4.3460546, -2.6508939, -4.3596959, -2.6616101, -0.9599385, 0.9836991
7: -15.4977894, -13.4751034, -15.4871826, -13.4763575, -1.2642879, 1.2576418
8: -0.5079880, 0.6885033, -0.5168147, 0.6970425, -0.6801038, 0.6818745
9: -6.4700294, -5.3437219, -6.4631238, -5.3138466, -0.7105472, 0.6868737

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4171804
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4147143
time: 3.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.8505535, -10.9745131, -12.8625689, -10.9683571, -1.0095012, 1.0557010
1: -10.9168215, -8.8665180, -10.8993158, -8.8819847, -1.2391353, 1.3125482
2: -10.5052338, -8.9651794, -10.4743452, -8.9141016, -1.2287693, 1.1165967
3: -4.1616750, -2.7165349, -4.1790810, -2.7228358, -0.8443251, 0.8498378
4: -14.8750801, -13.0515079, -14.8833008, -13.0229082, -1.0241537, 1.0117836
5: 8.5619984, 9.5777102, 8.5423536, 9.5638933, -0.6307567, 0.6570933
6: -4.3463135, -2.6448574, -4.3847599, -2.6418266, -0.9692166, 1.0298851
7: -15.4986229, -13.4755478, -15.4945955, -13.4748783, -1.2674122, 1.2646639
8: -0.5122340, 0.6885350, -0.5312014, 0.7171400, -0.7028699, 0.6985188
9: -6.4689784, -5.3432822, -6.4617753, -5.3115335, -0.7309072, 0.6903667

Time for backsubstitution: 8.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4180293
time: 2.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4155632
time: 3.01 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.8434849, -10.9667587, -12.8413811, -10.9713058, -1.0202777, 1.0032334
1: -10.8951416, -8.8618765, -10.8923025, -8.8701820, -1.2264838, 1.2558851
2: -10.4701309, -8.9559307, -10.4430666, -8.9480572, -1.1713033, 1.0912573
3: -4.1801362, -2.7246504, -4.1842451, -2.7258739, -0.8399053, 0.8681281
4: -14.8857985, -12.9814911, -14.8857307, -12.9820175, -1.0688376, 1.0106137
5: 8.5384216, 9.5710888, 8.5359993, 9.5607033, -0.6253245, 0.6647911
6: -4.3691792, -2.6469197, -4.3697824, -2.6616075, -0.9595850, 1.0001111
7: -15.4893475, -13.4698277, -15.4871807, -13.4727087, -1.2606306, 1.2716215
8: -0.5396793, 0.6971314, -0.5292799, 0.6970441, -0.6792848, 0.7038279
9: -6.4672656, -5.2989430, -6.4631233, -5.2940979, -0.7590947, 0.6717205

Time for backsubstitution: 8.47 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4130577, upper bound: 0.4174890
time: 2.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4130577, upper bound: 0.4153707
time: 2.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.8442307, -10.9626713, -12.8625755, -10.9625549, -1.0266752, 1.0504773
1: -10.8961811, -8.8695059, -10.8993196, -8.8810320, -1.2249746, 1.3224974
2: -10.4815445, -8.9549332, -10.4743462, -8.9036751, -1.2547441, 1.1050131
3: -4.1805201, -2.7242727, -4.1890955, -2.7228308, -0.8428140, 0.8713551
4: -14.8858223, -12.9841232, -14.8832989, -12.9876633, -1.0712276, 1.0153167
5: 8.5383177, 9.5715885, 8.5266762, 9.5638943, -0.6338664, 0.6789334
6: -4.3694344, -2.6408827, -4.3948469, -2.6418238, -0.9689798, 1.0463018
7: -15.4902210, -13.4702682, -15.4945965, -13.4712448, -1.2637844, 1.2785499
8: -0.5439341, 0.6971641, -0.5436671, 0.7171407, -0.7020612, 0.7204738
9: -6.4662147, -5.2985067, -6.4617753, -5.2917848, -0.7794554, 0.6752135

Time for backsubstitution: 8.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4162178, upper bound: 0.4183380
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4162178, upper bound: 0.4162197
time: 2.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.8498096, -10.9785995, -12.9637556, -10.9540405, -1.0235734, 1.1362457
1: -10.9157867, -8.8588867, -11.0010891, -8.8931217, -1.2546372, 1.3944840
2: -10.4938192, -8.9662266, -10.4862576, -8.9701271, -1.1471620, 1.1697693
3: -4.1612740, -2.7169170, -4.3066735, -2.7368939, -0.8628120, 1.0423353
4: -14.8750553, -13.0488749, -14.8876791, -12.9490643, -1.0999627, 1.0170839
5: 8.5621004, 9.5772228, 8.5427952, 9.5266609, -0.6398594, 0.6960888
6: -4.3460546, -2.6508939, -4.4040461, -2.6726327, -0.9663978, 1.0591555
7: -15.4977894, -13.4751034, -15.5284920, -13.5177937, -1.2598047, 1.3272810
8: -0.5079880, 0.6885033, -0.4893105, 0.7452798, -0.7866111, 0.7061779
9: -6.4700294, -5.3437219, -6.4604692, -5.2679105, -0.7660789, 0.6883404

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3957316, upper bound: 0.4096959
time: 3.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3957316, upper bound: 0.4072299
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.8505535, -10.9745131, -12.9850712, -10.9452896, -1.0299616, 1.1835499
1: -10.9168215, -8.8665180, -11.0082388, -8.9039669, -1.2531228, 1.4617791
2: -10.5052338, -8.9651794, -10.5175104, -8.9255390, -1.2310138, 1.1835799
3: -4.1616750, -2.7165349, -4.3115692, -2.7338181, -0.8656516, 1.0457196
4: -14.8750801, -13.0515079, -14.8852463, -12.9547081, -1.1024680, 1.0221775
5: 8.5619984, 9.5777102, 8.5334568, 9.5298319, -0.6483994, 0.7111084
6: -4.3463135, -2.6448574, -4.4296155, -2.6528783, -0.9757147, 1.1056347
7: -15.4986229, -13.4755478, -15.5361376, -13.5162754, -1.2630486, 1.3348393
8: -0.5122340, 0.6885350, -0.5036125, 0.7654667, -0.8095250, 0.7226067
9: -6.4689784, -5.3432822, -6.4591236, -5.2655802, -0.7866073, 0.6918337

Time for backsubstitution: 8.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3988918, upper bound: 0.4105449
time: 3.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3988918, upper bound: 0.4080789
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.8434849, -10.9667587, -12.9637623, -10.9482374, -1.0407372, 1.1309938
1: -10.8951416, -8.8618765, -11.0010967, -8.8921719, -1.2404699, 1.4044967
2: -10.4701309, -8.9559307, -10.4862585, -8.9596939, -1.1731844, 1.1582277
3: -4.1801362, -2.7246504, -4.3166871, -2.7368884, -0.8613749, 1.0638900
4: -14.8857985, -12.9814911, -14.8876801, -12.9138145, -1.1470542, 1.0206671
5: 8.5384216, 9.5710888, 8.5270433, 9.5266628, -0.6430798, 0.7179658
6: -4.3691792, -2.6469197, -4.4141335, -2.6726313, -0.9660668, 1.0755692
7: -15.4893475, -13.4698277, -15.5284920, -13.5141506, -1.2561440, 1.3413234
8: -0.5396793, 0.6971314, -0.5017762, 0.7452807, -0.7858038, 0.7281356
9: -6.4672656, -5.2989430, -6.4604692, -5.2481637, -0.8146267, 0.6731870

Time for backsubstitution: 8.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4007892, upper bound: 0.4100047
time: 3.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4007892, upper bound: 0.4078863
time: 3.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.8442307, -10.9626713, -12.9850826, -10.9394875, -1.0471354, 1.1783803
1: -10.8961811, -8.8695059, -11.0082455, -8.9030142, -1.2389612, 1.4717374
2: -10.4815445, -8.9549332, -10.5175114, -8.9151115, -1.2570062, 1.1719935
3: -4.1805201, -2.7242727, -4.3215833, -2.7338140, -0.8642097, 1.0672507
4: -14.8858223, -12.9841232, -14.8852501, -12.9194565, -1.1495466, 1.0257764
5: 8.5383177, 9.5715885, 8.5177164, 9.5298319, -0.6515505, 0.7329993
6: -4.3694344, -2.6408827, -4.4397039, -2.6528778, -0.9755013, 1.1220522
7: -15.4902210, -13.4702682, -15.5361385, -13.5126438, -1.2594175, 1.3487825
8: -0.5439341, 0.6971641, -0.5160785, 0.7654674, -0.8087273, 0.7445614
9: -6.4662147, -5.2985067, -6.4591236, -5.2458324, -0.8351555, 0.6766803

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4039493, upper bound: 0.4108521
time: 3.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4039493, upper bound: 0.4087339
time: 3.08 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.9637556, -10.9540405, -12.8498096, -10.9785995, -1.1362462, 1.0235734
1: -11.0010891, -8.8931217, -10.9157867, -8.8588867, -1.3944840, 1.2546372
2: -10.4862576, -8.9701271, -10.4938192, -8.9662266, -1.1697693, 1.1471622
3: -4.3066735, -2.7368939, -4.1612740, -2.7169170, -1.0423353, 0.8628120
4: -14.8876791, -12.9490643, -14.8750553, -13.0488749, -1.0170836, 1.0999625
5: 8.5427952, 9.5266609, 8.5621004, 9.5772228, -0.6960888, 0.6398594
6: -4.4040461, -2.6726327, -4.3460546, -2.6508939, -1.0591550, 0.9663978
7: -15.5284920, -13.5177937, -15.4977894, -13.4751034, -1.3272810, 1.2598047
8: -0.4893105, 0.7452798, -0.5079880, 0.6885033, -0.7061778, 0.7866111
9: -6.4604692, -5.2679105, -6.4700294, -5.3437219, -0.6883404, 0.7660789

Time for backsubstitution: 9.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3943290
time: 3.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3957330
time: 3.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.9637556, -10.9540405, -12.9723043, -10.9555321, -1.0489366, 1.0435553
1: -11.0010891, -8.8931217, -11.0247440, -8.8808718, -1.2675967, 1.2623754
2: -10.4862576, -8.9701271, -10.5370197, -8.9779282, -1.1047559, 1.1473222
3: -4.3066735, -2.7368939, -4.2937455, -2.7279284, -0.8720517, 0.8668146
4: -14.8876791, -12.9490643, -14.8770084, -12.9806786, -1.0188494, 1.0335734
5: 8.5427952, 9.5266609, 8.5532103, 9.5431843, -0.6311748, 0.6103067
6: -4.4040461, -2.6726327, -4.3902664, -2.6619105, -0.9859610, 0.9621079
7: -15.5284920, -13.5177937, -15.5395041, -13.5165567, -1.2641234, 1.2708127
8: -0.4893105, 0.7452798, -0.4805288, 0.7367535, -0.6899301, 0.6881706
9: -6.4604692, -5.2679105, -6.4673777, -5.2977619, -0.6975443, 0.7211744

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3943289
time: 3.26 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3957330
time: 3.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.9850712, -10.9452896, -12.8505535, -10.9745131, -1.1835494, 1.0299616
1: -11.0082388, -8.9039669, -10.9168215, -8.8665180, -1.4617791, 1.2531226
2: -10.5175104, -8.9255390, -10.5052338, -8.9651794, -1.1835799, 1.2310138
3: -4.3115692, -2.7338181, -4.1616750, -2.7165349, -1.0457196, 0.8656520
4: -14.8852463, -12.9547081, -14.8750801, -13.0515079, -1.0221775, 1.1024680
5: 8.5334568, 9.5298319, 8.5619984, 9.5777102, -0.7111084, 0.6483994
6: -4.4296155, -2.6528783, -4.3463135, -2.6448574, -1.1056345, 0.9757147
7: -15.5361376, -13.5162754, -15.4986229, -13.4755478, -1.3348393, 1.2630482
8: -0.5036125, 0.7654667, -0.5122340, 0.6885350, -0.7226069, 0.8095250
9: -6.4591236, -5.2655802, -6.4689784, -5.3432822, -0.6918337, 0.7866075

Time for backsubstitution: 9.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3974891
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3988931
time: 2.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.9850712, -10.9452896, -12.9730759, -10.9514465, -1.0963221, 1.0499394
1: -11.0082388, -8.9039669, -11.0258465, -8.8884964, -1.3345737, 1.2608562
2: -10.5175104, -8.9255390, -10.5484352, -8.9768791, -1.1185031, 1.2309406
3: -4.3115692, -2.7338181, -4.2941561, -2.7275493, -0.8753176, 0.8697325
4: -14.8852463, -12.9547081, -14.8770342, -12.9833069, -1.0238988, 1.0359886
5: 8.5334568, 9.5298319, 8.5531082, 9.5436726, -0.6461933, 0.6188911
6: -4.4296155, -2.6528783, -4.3905044, -2.6558719, -1.0324326, 0.9713783
7: -15.5361376, -13.5162754, -15.5404749, -13.5170021, -1.2711387, 1.2739105
8: -0.5036125, 0.7654667, -0.4847741, 0.7367868, -0.7065929, 0.7110927
9: -6.4591236, -5.2655802, -6.4663262, -5.2973104, -0.7010629, 0.7416320

Time for backsubstitution: 9.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3974890
time: 2.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3988931
time: 3.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.9637623, -10.9482374, -12.8434849, -10.9667587, -1.1309936, 1.0407372
1: -11.0010967, -8.8921719, -10.8951416, -8.8618765, -1.4044967, 1.2404702
2: -10.4862585, -8.9596939, -10.4701309, -8.9559307, -1.1582274, 1.1731842
3: -4.3166871, -2.7368884, -4.1801362, -2.7246504, -1.0638895, 0.8613750
4: -14.8876801, -12.9138145, -14.8857985, -12.9814911, -1.0206671, 1.1470542
5: 8.5270433, 9.5266628, 8.5384216, 9.5710888, -0.7179660, 0.6430798
6: -4.4141335, -2.6726313, -4.3691792, -2.6469197, -1.0755692, 0.9660668
7: -15.5284920, -13.5141506, -15.4893475, -13.4698277, -1.3413229, 1.2561438
8: -0.5017762, 0.7452807, -0.5396793, 0.6971314, -0.7281356, 0.7858036
9: -6.4604692, -5.2481637, -6.4672656, -5.2989430, -0.6731870, 0.8146262

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4070010
time: 3.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4000244
time: 3.02 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.9637623, -10.9482374, -12.9659405, -10.9436913, -1.0436838, 1.0606651
1: -11.0010967, -8.8921719, -11.0041056, -8.8838634, -1.2776079, 1.2481995
2: -10.4862585, -8.9596939, -10.5133305, -8.9675684, -1.0932152, 1.1733444
3: -4.3166871, -2.7368884, -4.3126230, -2.7356641, -0.8935380, 0.8653765
4: -14.8876801, -12.9138145, -14.8877449, -12.9132929, -1.0224333, 1.0806003
5: 8.5270433, 9.5266628, 8.5294609, 9.5370321, -0.6530089, 0.6135274
6: -4.4141335, -2.6726313, -4.4134355, -2.6579356, -1.0023527, 0.9617751
7: -15.5284920, -13.5141506, -15.5310068, -13.5112810, -1.2781649, 1.2670939
8: -0.5017762, 0.7452807, -0.5122106, 0.7453732, -0.7118769, 0.6873624
9: -6.4604692, -5.2481637, -6.4646130, -5.2529764, -0.6823916, 0.7697225

Time for backsubstitution: 9.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4070008
time: 3.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4000244
time: 3.12 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.9850826, -10.9394875, -12.8442307, -10.9626713, -1.1783800, 1.0471354
1: -11.0082455, -8.9030142, -10.8961811, -8.8695059, -1.4717374, 1.2389612
2: -10.5175114, -8.9151115, -10.4815445, -8.9549332, -1.1719933, 1.2570062
3: -4.3215833, -2.7338140, -4.1805201, -2.7242727, -1.0672507, 0.8642101
4: -14.8852501, -12.9194565, -14.8858223, -12.9841232, -1.0257764, 1.1495469
5: 8.5177164, 9.5298319, 8.5383177, 9.5715885, -0.7329992, 0.6515504
6: -4.4397039, -2.6528778, -4.3694344, -2.6408827, -1.1220522, 0.9755011
7: -15.5361385, -13.5126438, -15.4902210, -13.4702682, -1.3487830, 1.2594173
8: -0.5160785, 0.7654674, -0.5439341, 0.6971641, -0.7445612, 0.8087273
9: -6.4591236, -5.2458324, -6.4662147, -5.2985067, -0.6766803, 0.8351552

Time for backsubstitution: 8.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4101610
time: 3.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4031845
time: 3.15 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.9850826, -10.9394875, -12.9667168, -10.9396057, -1.0911517, 1.0670588
1: -11.0082455, -8.9030142, -11.0052052, -8.8914881, -1.3445320, 1.2466855
2: -10.5175114, -8.9151115, -10.5247459, -8.9665680, -1.1069169, 1.2569332
3: -4.3215833, -2.7338140, -4.3130193, -2.7352858, -0.8967795, 0.8682909
4: -14.8852501, -12.9194565, -14.8877716, -12.9159222, -1.0274968, 1.0830016
5: 8.5177164, 9.5298319, 8.5293541, 9.5375338, -0.6680415, 0.6220419
6: -4.4397039, -2.6528778, -4.4136705, -2.6518979, -1.0488284, 0.9711637
7: -15.5361385, -13.5126438, -15.5320234, -13.5117226, -1.2850828, 1.2702212
8: -0.5160785, 0.7654674, -0.5164649, 0.7454093, -0.7285368, 0.7102957
9: -6.4591236, -5.2458324, -6.4635611, -5.2525272, -0.6859107, 0.7901793

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4101609
time: 3.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4031845
time: 3.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.04 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4171804
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4147143
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4180293
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4155632
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4130577, upper bound: 0.4174890
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4130577, upper bound: 0.4153707
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4162178, upper bound: 0.4183380
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4162178, upper bound: 0.4162197
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.3957316, upper bound: 0.4096959
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.3957316, upper bound: 0.4072299
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.3988918, upper bound: 0.4105449
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.3988918, upper bound: 0.4080789
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4007892, upper bound: 0.4100047
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4007892, upper bound: 0.4078863
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4039493, upper bound: 0.4108521
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4039493, upper bound: 0.4087339
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3943290
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3957330
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3943289
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4024443, upper bound: 0.3957330
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3974891
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3988931
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3974890
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4032933, upper bound: 0.3988931
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4070010
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4000244
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4070008
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4023342, upper bound: 0.4000244
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4101610
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4031845
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4101609
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 5, lower bound: -0.4031832, upper bound: 0.4031845

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.8463888, -10.9796219, -12.8409319, -10.9771080, -0.9967380, 1.0027411
1: -10.8520679, -8.8919325, -10.8713303, -8.8711367, -1.1559567, 1.1386573
2: -10.4806643, -8.9791069, -10.4389772, -8.9590807, -1.1269689, 1.0808446
3: -4.1614499, -2.7312691, -4.1741538, -2.7296600, -0.8168981, 0.8282363
4: -14.8629608, -13.0443401, -14.8828001, -13.0173206, -0.9962959, 0.9847105
5: 8.5735140, 9.5622311, 8.5519361, 9.5583601, -0.5978305, 0.6220400
6: -4.3448100, -2.6513481, -4.3595638, -2.6616595, -0.9577191, 0.9824004
7: -15.4683485, -13.4941483, -15.4811268, -13.4764061, -1.2297401, 1.2198029
8: -0.4766481, 0.6647196, -0.5060318, 0.6969864, -0.6553288, 0.6388183
9: -6.4690433, -5.3461657, -6.4631233, -5.3144360, -0.7089198, 0.6842258

Time for backsubstitution: 9.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4088442
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4171803
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.8495684, -10.9785995, -12.8413010, -10.9771080, -0.9999495, 1.0057349
1: -10.8952618, -8.8588867, -10.8845968, -8.8711367, -1.1039593, 1.2429609
2: -10.4877892, -8.9663754, -10.4400883, -8.9585419, -1.1559362, 1.0853255
3: -4.1612606, -2.7319481, -4.1742244, -2.7311430, -0.8329742, 0.8334619
4: -14.8607235, -13.0488901, -14.8793278, -13.0172729, -0.9967518, 0.9998286
5: 8.5624294, 9.5725336, 8.5517883, 9.5572653, -0.6195014, 0.6208532
6: -4.3459268, -2.6509409, -4.3596559, -2.6616235, -0.9594772, 0.9833119
7: -15.4914856, -13.4751577, -15.4827499, -13.4763718, -1.2291460, 1.2558661
8: -0.5030599, 0.6884308, -0.5153053, 0.6970220, -0.6351900, 0.6802573
9: -6.4700308, -5.3444419, -6.4631238, -5.3139663, -0.7103624, 0.6850009

Time for backsubstitution: 8.62 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4085389
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4080001, upper bound: 0.4147143
time: 2.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.8471355, -10.9755373, -12.8621101, -10.9683571, -1.0031168, 1.0499020
1: -10.8531361, -8.8995647, -10.8783588, -8.8819847, -1.1544666, 1.2053282
2: -10.4920807, -8.9781265, -10.4702597, -8.9147577, -1.2103963, 1.0945315
3: -4.1618395, -2.7308879, -4.1789956, -2.7266183, -0.8197727, 0.8314680
4: -14.8629856, -13.0469666, -14.8803692, -13.0229702, -0.9986935, 0.9893694
5: 8.5734091, 9.5627460, 8.5426044, 9.5615520, -0.6064472, 0.6361527
6: -4.3450732, -2.6453133, -4.3846111, -2.6418810, -0.9669936, 1.0285776
7: -15.4692364, -13.4945908, -15.4885607, -13.4749317, -1.2328711, 1.2268052
8: -0.4808903, 0.6647534, -0.5204604, 0.7170811, -0.6781058, 0.6554642
9: -6.4679928, -5.3457108, -6.4617753, -5.3121181, -0.7292774, 0.6877189

Time for backsubstitution: 8.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4096932
time: 3.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4180292
time: 3.03 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.8503084, -10.9745131, -12.8624916, -10.9683571, -1.0063388, 1.0528696
1: -10.8963032, -8.8665180, -10.8916197, -8.8819847, -1.1024780, 1.3095951
2: -10.4992065, -8.9653339, -10.4713669, -8.9141617, -1.2391400, 1.0991879
3: -4.1616602, -2.7315640, -4.1790762, -2.7281005, -0.8358998, 0.8366268
4: -14.8607473, -13.0515223, -14.8768959, -13.0229187, -0.9991202, 1.0045130
5: 8.5623274, 9.5730190, 8.5424557, 9.5604477, -0.6281122, 0.6349933
6: -4.3461876, -2.6449072, -4.3847141, -2.6418419, -0.9687538, 1.0294840
7: -15.4923306, -13.4756021, -15.4901810, -13.4748926, -1.2322407, 1.2628410
8: -0.5073063, 0.6884627, -0.5296922, 0.7171173, -0.6579552, 0.6968613
9: -6.4689784, -5.3440022, -6.4617753, -5.3116517, -0.7307160, 0.6885052

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4093878
time: 3.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111603, upper bound: 0.4155633
time: 2.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.8399687, -10.9666119, -12.8409405, -10.9713058, -1.0136514, 0.9974670
1: -10.8327723, -8.8944759, -10.8713341, -8.8701820, -1.1608186, 1.1450067
2: -10.4572487, -8.9598913, -10.4389791, -8.9486542, -1.1539896, 1.0680044
3: -4.1907506, -2.7361457, -4.1841679, -2.7296546, -0.8109272, 0.8565249
4: -14.8765125, -12.9638119, -14.8828030, -12.9820719, -1.0575013, 0.9821851
5: 8.5343046, 9.5635376, 8.5362511, 9.5583601, -0.5980017, 0.6540096
6: -4.3685389, -2.6474159, -4.3696523, -2.6616588, -0.9573002, 0.9984229
7: -15.4689064, -13.4739466, -15.4811287, -13.4727631, -1.2370920, 1.2300432
8: -0.5076718, 0.6729319, -0.5186810, 0.6969883, -0.6544797, 0.6605685
9: -6.4660549, -5.3018546, -6.4631238, -5.2946887, -0.7565141, 0.6690952

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4124013, upper bound: 0.4088442
time: 3.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4124015, upper bound: 0.4095221
time: 3.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.8432474, -10.9667587, -12.8413048, -10.9713058, -1.0169475, 1.0026009
1: -10.8746243, -8.8618765, -10.8846073, -8.8701820, -1.1138196, 1.2554123
2: -10.4598608, -8.9560852, -10.4400911, -8.9481115, -1.1886063, 1.0766056
3: -4.1801195, -2.7426977, -4.1842384, -2.7311382, -0.8345940, 0.8678074
4: -14.8637056, -12.9815044, -14.8793268, -12.9820194, -1.0668564, 1.0076382
5: 8.5387449, 9.5599651, 8.5360985, 9.5572653, -0.6250057, 0.6545339
6: -4.3690538, -2.6469674, -4.3697462, -2.6616228, -0.9591472, 0.9998074
7: -15.4777985, -13.4698820, -15.4827480, -13.4727249, -1.2309113, 1.2712555
8: -0.5349097, 0.6970620, -0.5277712, 0.6970239, -0.6344599, 0.7024038
9: -6.4672651, -5.2993407, -6.4631238, -5.2942185, -0.7587900, 0.6698550

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4124013, upper bound: 0.4085389
time: 3.27 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4124015, upper bound: 0.4098651
time: 3.04 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.18 + 546.85 = 604.03 seconds
