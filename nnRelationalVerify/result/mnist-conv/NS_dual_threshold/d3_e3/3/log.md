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
execution time: IAR + RelationalAnalysis = 23.39 + 34.36 = 57.74 seconds
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
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4243539
time: 3.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4195697
time: 2.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.25 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.25
Output dim: 5, lower bound: -0.4195681, upper bound: 0.4243539
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.25
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

Time for backsubstitution: 8.62 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4182399
time: 2.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4190889
time: 3.35 seconds

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

Time for backsubstitution: 8.06 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4134558
time: 2.86 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4143048
time: 3.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.18
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4182399
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.18
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4190889
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.18
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4134558
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.18
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4143048

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.8434973, -10.9617138, -12.8418140, -10.9348412, -1.0556598, 1.0336788
1: -10.8951616, -8.8540678, -10.8928347, -8.8319359, -1.2559104, 1.2506781
2: -10.4701414, -8.9441662, -10.4430752, -8.9313469, -1.1886780, 1.1388750
3: -4.1862659, -2.7246363, -4.1853895, -2.6810970, -0.8967214, 0.8687959
4: -14.8858013, -12.9779425, -14.9114647, -12.9815168, -1.0689964, 1.0877247
5: 8.5352058, 9.5710897, 8.5347500, 9.5707951, -0.6753805, 0.6672660
6: -4.3709326, -2.6469176, -4.3707800, -2.6434879, -1.0026722, 1.0009551
7: -15.4893494, -13.4659672, -15.4873657, -13.4539118, -1.2842417, 1.2669358
8: -0.5410397, 0.6971312, -0.5466025, 0.6972928, -0.7141073, 0.7294195
9: -6.4672656, -5.2920656, -6.4785423, -5.2931685, -0.7591844, 0.7605379

Time for backsubstitution: 7.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4182400
time: 3.06 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4182399
time: 2.82 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.8442421, -10.9576292, -12.8629837, -10.9260912, -1.0620575, 1.0807998
1: -10.8962030, -8.8616962, -10.8998356, -8.8427858, -1.2544012, 1.3172863
2: -10.4815588, -8.9431677, -10.4743567, -8.8869858, -1.2720637, 1.1526639
3: -4.1866550, -2.7242577, -4.1902361, -2.6780410, -0.8995867, 0.8720198
4: -14.8858280, -12.9805746, -14.9090319, -12.9871607, -1.0713854, 1.0923615
5: 8.5350981, 9.5715923, 8.5254574, 9.5739841, -0.6838888, 0.6812520
6: -4.3711929, -2.6408820, -4.3957705, -2.6237073, -1.0119658, 1.0470946
7: -15.4902267, -13.4664059, -15.4947824, -13.4524345, -1.2875161, 1.2739539
8: -0.5452859, 0.6971655, -0.5609961, 0.7173719, -0.7368519, 0.7458947
9: -6.4662151, -5.2916284, -6.4771957, -5.2908592, -0.7795408, 0.7640390

Time for backsubstitution: 7.71 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4190889
time: 2.87 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4190889
time: 3.06 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.9659538, -10.9386482, -12.8407516, -10.9401884, -1.1774397, 1.0737872
1: -11.0041246, -8.8760529, -10.8915596, -8.8555794, -1.3960781, 1.2730029
2: -10.5133448, -8.9558020, -10.4430475, -8.9418802, -1.2505379, 1.1436932
3: -4.3187461, -2.7356503, -4.1851940, -2.6996121, -1.0900598, 0.8939805
4: -14.8877459, -12.9097404, -14.9027863, -12.9815912, -1.0807338, 1.1635413
5: 8.5262661, 9.5370331, 8.5366001, 9.5558653, -0.7292905, 0.6604220
6: -4.4151917, -2.6579351, -4.3689747, -2.6539633, -1.0721831, 1.0060303
7: -15.5310097, -13.5074215, -15.4869232, -13.4764099, -1.3431287, 1.2728274
8: -0.5135772, 0.7453725, -0.5269694, 0.6967127, -0.7238712, 0.8358884
9: -6.4646134, -5.2460966, -6.4721227, -5.2934093, -0.7693090, 0.8123472

Time for backsubstitution: 7.67 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4015361, upper bound: 0.4093111
time: 4.39 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4069994, upper bound: 0.4093123
time: 2.86 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.9667301, -10.9345608, -12.8619900, -10.9314384, -1.1838760, 1.1210029
1: -11.0052223, -8.8836794, -10.8986006, -8.8664284, -1.3947268, 1.3397431
2: -10.5247602, -8.9548025, -10.4743242, -8.8974676, -1.3340020, 1.1574292
3: -4.3191490, -2.7352719, -4.1900511, -2.6965516, -1.0929499, 0.8972113
4: -14.8877735, -12.9123716, -14.9003525, -12.9872322, -1.0831265, 1.1683831
5: 8.5261583, 9.5375347, 8.5272875, 9.5590496, -0.7377572, 0.6747499
6: -4.4154272, -2.6518960, -4.3941393, -2.6341846, -1.0814836, 1.0522807
7: -15.5320234, -13.5078640, -15.4943314, -13.4749098, -1.3466492, 1.2798481
8: -0.5178225, 0.7454083, -0.5413270, 0.7168331, -0.7466848, 0.8523142
9: -6.4635615, -5.2456503, -6.4707756, -5.2910919, -0.7896757, 0.8159118

Time for backsubstitution: 8.27 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4046962, upper bound: 0.4101612
time: 3.17 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4101613
time: 2.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.48 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4182400
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4111430, upper bound: 0.4182399
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4190889
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4143031, upper bound: 0.4190889
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4015361, upper bound: 0.4093111
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4069994, upper bound: 0.4093123
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4046962, upper bound: 0.4101612
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.48
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4101613

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -12.8434973, -10.9617138, -12.8413811, -10.9705715, -1.0204916, 1.0332417
1: -10.8951616, -8.8540678, -10.8923073, -8.8692417, -1.2274556, 1.2502306
2: -10.4701414, -8.9441662, -10.4430695, -8.9466419, -1.1722391, 1.1384220
3: -4.1862659, -2.7246363, -4.1853099, -2.7258747, -0.8683739, 0.8687510
4: -14.8858013, -12.9779425, -14.8857355, -12.9815474, -1.0689774, 1.0723994
5: 8.5352058, 9.5710897, 8.5355434, 9.5607042, -0.6572068, 0.6649867
6: -4.3709326, -2.6469176, -4.3699937, -2.6616073, -0.9816282, 1.0002780
7: -15.4893494, -13.4659672, -15.4871836, -13.4722471, -1.2611790, 1.2669010
8: -0.5410397, 0.6971312, -0.5294695, 0.6970448, -0.7135358, 0.7039651
9: -6.4672656, -5.2920656, -6.4631233, -5.2932692, -0.7591255, 0.7525470

Time for backsubstitution: 8.49 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4015361, upper bound: 0.4140964
time: 3.08 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4069994, upper bound: 0.4140953
time: 4.76 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -12.8434973, -10.9617138, -12.9637661, -10.9475060, -1.0409517, 1.1609473
1: -10.8951616, -8.8540678, -11.0010977, -8.8912287, -1.2414432, 1.3988309
2: -10.4701414, -8.9441662, -10.4862614, -8.9582767, -1.1741204, 1.2053952
3: -4.1862659, -2.7246363, -4.3177533, -2.7368884, -0.8897729, 1.0645132
4: -14.8858013, -12.9779425, -14.8876801, -12.9133406, -1.1471949, 1.0823863
5: 8.5352058, 9.5710897, 8.5266066, 9.5266628, -0.6749184, 0.7181611
6: -4.3709326, -2.6469176, -4.4143472, -2.6726315, -0.9880877, 1.0757360
7: -15.4893494, -13.4659672, -15.5284948, -13.5136833, -1.2566924, 1.3365412
8: -0.5410397, 0.6971312, -0.5019634, 0.7452803, -0.8200436, 0.7282677
9: -6.4672656, -5.2920656, -6.4604692, -5.2473359, -0.8146572, 0.7540135

Time for backsubstitution: 9.29 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4069995, upper bound: 0.4086329
time: 3.35 seconds

## Relational analysis of NS_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4069995, upper bound: 0.4140964
time: 3.29 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -12.8442421, -10.9576292, -12.8625765, -10.9618225, -1.0268903, 1.0804019
1: -10.8962030, -8.8616962, -10.8993225, -8.8800907, -1.2259464, 1.3168955
2: -10.4815588, -8.9431677, -10.4743481, -8.9022579, -1.2556791, 1.1522100
3: -4.1866550, -2.7242577, -4.1901603, -2.7228293, -0.8712740, 0.8719778
4: -14.8858280, -12.9805746, -14.8833036, -12.9871902, -1.0713677, 1.0770802
5: 8.5350981, 9.5715923, 8.5262423, 9.5638943, -0.6658182, 0.6791295
6: -4.3711929, -2.6408820, -4.3950591, -2.6418254, -0.9909072, 1.0464680
7: -15.4902267, -13.4664059, -15.4945965, -13.4707804, -1.2643323, 1.2739229
8: -0.5452859, 0.6971655, -0.5438540, 0.7171419, -0.7363071, 0.7206111
9: -6.4662151, -5.2916284, -6.4617753, -5.2909560, -0.7794864, 0.7560477

Time for backsubstitution: 8.90 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4046962, upper bound: 0.4149454
time: 3.03 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4149454
time: 3.03 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -12.8442421, -10.9576292, -12.9850826, -10.9387550, -1.0473490, 1.2082500
1: -10.8962030, -8.8616962, -11.0082455, -8.9020710, -1.2399335, 1.4661264
2: -10.4815588, -8.9431677, -10.5175142, -8.9136934, -1.2579408, 1.2191935
3: -4.1866550, -2.7242577, -4.3226480, -2.7338114, -0.8925996, 1.0678740
4: -14.8858280, -12.9805746, -14.8852501, -12.9189873, -1.1496868, 1.0874739
5: 8.5350981, 9.5715923, 8.5173044, 9.5298328, -0.6834610, 0.7331948
6: -4.3711929, -2.6408820, -4.4399185, -2.6528773, -0.9974051, 1.1222196
7: -15.4902267, -13.4664059, -15.5361385, -13.5121775, -1.2599640, 1.3440986
8: -0.5452859, 0.6971655, -0.5162663, 0.7654674, -0.8429618, 0.7446985
9: -6.4662151, -5.2916284, -6.4591236, -5.2450037, -0.8351867, 0.7575147

Time for backsubstitution: 9.31 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4094819
time: 2.76 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4149454
time: 3.08 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.9723043, -10.9555321, -12.8407421, -10.9467239, -1.1601143, 1.0490866
1: -11.0247440, -8.8808718, -10.8915520, -8.8574734, -1.4092798, 1.2686563
2: -10.5370197, -8.9779282, -10.4430437, -8.9537287, -1.2235928, 1.1080492
3: -4.2937455, -2.7279284, -4.1741142, -2.6996195, -1.0630834, 0.8718830
4: -14.8770084, -12.9806786, -14.9027843, -13.0173101, -1.0335712, 1.0982258
5: 8.5532103, 9.5431843, 8.5527525, 9.5558662, -0.6941662, 0.6384318
6: -4.3902664, -2.6619105, -4.3586755, -2.6539643, -1.0504181, 0.9894731
7: -15.5395041, -13.5165567, -15.4869204, -13.4805193, -1.3462977, 1.2635713
8: -0.4805288, 0.7367535, -0.5143151, 0.6967103, -0.6904480, 0.8138099
9: -6.4673777, -5.2977619, -6.4721227, -5.3139858, -0.7207303, 0.7466755

Time for backsubstitution: 9.37 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3982197, upper bound: 0.4000506
time: 3.70 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3982197, upper bound: 0.4059958
time: 3.24 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.9659405, -10.9436913, -12.8407478, -10.9409218, -1.1772261, 1.0437946
1: -11.0041056, -8.8838634, -10.8915586, -8.8565178, -1.3951058, 1.2786596
2: -10.5133305, -8.9675684, -10.4430456, -8.9432964, -1.2496023, 1.0965104
3: -4.3126230, -2.7356641, -4.1841283, -2.6996140, -1.0615914, 0.8933578
4: -14.8877449, -12.9132929, -14.9027843, -12.9820604, -1.0805931, 1.1017590
5: 8.5294609, 9.5370321, 8.5370502, 9.5558653, -0.6973538, 0.6602259
6: -4.4134355, -2.6579356, -4.3687658, -2.6539643, -1.0500727, 1.0058637
7: -15.5310068, -13.5112810, -15.4869213, -13.4768734, -1.3425798, 1.2775657
8: -0.5122106, 0.7453732, -0.5267813, 0.6967120, -0.6896318, 0.8357515
9: -6.4646130, -5.2529764, -6.4721227, -5.2942371, -0.7692785, 0.7315228

Time for backsubstitution: 9.38 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4036830, upper bound: 0.4000493
time: 5.96 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4036830, upper bound: 0.4059943
time: 5.25 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.9730759, -10.9514465, -12.8619833, -10.9379730, -1.1665411, 1.0963027
1: -11.0258465, -8.8884964, -10.8985901, -8.8683224, -1.4079218, 1.3353972
2: -10.5484352, -8.9768791, -10.4743214, -8.9093113, -1.3070855, 1.1217961
3: -4.2941561, -2.7275493, -4.1789722, -2.6965580, -1.0659857, 0.8751373
4: -14.8770342, -12.9833069, -14.9003487, -13.0229549, -1.0359769, 1.1030741
5: 8.5531082, 9.5436726, 8.5434074, 9.5590477, -0.7026291, 0.6527461
6: -4.3905044, -2.6558719, -4.3838396, -2.6341891, -1.0597186, 1.0357189
7: -15.5404749, -13.5170021, -15.4943275, -13.4790068, -1.3497877, 1.2705913
8: -0.4847741, 0.7367868, -0.5286741, 0.7168307, -0.7132571, 0.8302338
9: -6.4663262, -5.2973104, -6.4707751, -5.3116689, -0.7410970, 0.7502308

Time for backsubstitution: 9.33 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4013798, upper bound: 0.4008997
time: 3.72 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4013798, upper bound: 0.4068447
time: 3.33 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.9667168, -10.9396057, -12.8619852, -10.9321709, -1.1836624, 1.0910943
1: -11.0052052, -8.8914881, -10.8985977, -8.8673687, -1.3937540, 1.3453472
2: -10.5247459, -8.9665680, -10.4743252, -8.8988848, -1.3330669, 1.1102123
3: -4.3130193, -2.7352858, -4.1889849, -2.6965532, -1.0644898, 0.8965881
4: -14.8877716, -12.9159222, -14.9003525, -12.9877014, -1.0829864, 1.1066217
5: 8.5293541, 9.5375338, 8.5277157, 9.5590496, -0.7057494, 0.6745548
6: -4.4136705, -2.6518979, -4.3939276, -2.6341853, -1.0594902, 1.0521140
7: -15.5320234, -13.5117226, -15.4943314, -13.4753714, -1.3461003, 1.2844913
8: -0.5164649, 0.7454093, -0.5411410, 0.7168329, -0.7124510, 0.8521771
9: -6.4635611, -5.2525272, -6.4707756, -5.2919211, -0.7896445, 0.7350783

Time for backsubstitution: 9.53 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4068431, upper bound: 0.4008986
time: 5.43 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4068431, upper bound: 0.4068437
time: 3.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 18.88 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4015361, upper bound: 0.4140964
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4069994, upper bound: 0.4140953
NS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4069995, upper bound: 0.4086329
NS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4069995, upper bound: 0.4140964
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4046962, upper bound: 0.4149454
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4149454
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4094819
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4101596, upper bound: 0.4149454
NS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.3982197, upper bound: 0.4000506
NS_A2_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.3982197, upper bound: 0.4059958
NS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4036830, upper bound: 0.4000493
NS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4036830, upper bound: 0.4059943
NS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4013798, upper bound: 0.4008997
NS_A2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4013798, upper bound: 0.4068447
NS_A2_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4068431, upper bound: 0.4008986
NS_A2_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 18.88
Output dim: 5, lower bound: -0.4068431, upper bound: 0.4068437

## BFS NS instance: NS_A1_B1_B1_A1

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

Time for backsubstitution: 9.47 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4113316, upper bound: 0.4108485
time: 3.27 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4114701, upper bound: 0.4192462
time: 3.14 seconds

## BFS NS instance: NS_A1_B1_B1_A2

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

Time for backsubstitution: 9.18 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4167950, upper bound: 0.4108485
time: 3.24 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4169335, upper bound: 0.4192463
time: 2.85 seconds

## BFS NS instance: NS_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -12.8434868, -10.9682484, -12.9700928, -10.9643888, -1.0162501, 1.1436369
1: -10.8951511, -8.8559637, -11.0217209, -8.8960485, -1.2370958, 1.4120369
2: -10.4701366, -8.9560070, -10.5099392, -8.9805136, -1.1384497, 1.1784480
3: -4.1751852, -2.7246423, -4.2927284, -2.7291780, -0.8677092, 1.0375054
4: -14.8857975, -13.0136623, -14.8769426, -12.9842834, -1.0818782, 1.0352378
5: 8.5513611, 9.5710888, 8.5535326, 9.5328522, -0.6530170, 0.6830466
6: -4.3606334, -2.6469200, -4.3894296, -2.6766047, -0.9715595, 1.0539911
7: -15.4893494, -13.4700794, -15.5370979, -13.5228186, -1.2474370, 1.3398252
8: -0.5283875, 0.6971316, -0.4689138, 0.7366619, -0.7979677, 0.6948671
9: -6.4672656, -5.3126411, -6.4632339, -5.2990041, -0.7490199, 0.7054348

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4036753, upper bound: 0.3984131
time: 3.07 seconds

## Relational analysis of NS_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4036753, upper bound: 0.4061549
time: 2.91 seconds

## BFS NS instance: NS_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -12.8434963, -10.9624481, -12.9637480, -10.9525480, -1.0109282, 1.1607335
1: -10.8951588, -8.8550100, -11.0010786, -8.8990393, -1.2470932, 1.3978591
2: -10.4701424, -8.9455833, -10.4862490, -8.9700413, -1.1269445, 1.2044592
3: -4.1851997, -2.7246373, -4.3116484, -2.7369020, -0.8891521, 1.0360034
4: -14.8857994, -12.9784126, -14.8876791, -12.9169006, -1.0853906, 1.0822456
5: 8.5356617, 9.5710888, 8.5297956, 9.5266609, -0.6747229, 0.6861386
6: -4.3707223, -2.6469188, -4.4125891, -2.6726308, -0.9879377, 1.0536027
7: -15.4893484, -13.4664345, -15.5284901, -13.5175438, -1.2612972, 1.3359923
8: -0.5408530, 0.6971316, -0.5005679, 0.7452803, -0.8199220, 0.6940403
9: -6.4672656, -5.2928934, -6.4604692, -5.2542129, -0.7338667, 0.7539821

Time for backsubstitution: 8.70 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4036753, upper bound: 0.4038766
time: 3.39 seconds

## Relational analysis of NS_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4036753, upper bound: 0.4116184
time: 3.15 seconds

## BFS NS instance: NS_A1_B2_B1_A1

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

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4144917, upper bound: 0.4116975
time: 3.29 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4146302, upper bound: 0.4200951
time: 3.69 seconds

## BFS NS instance: NS_A1_B2_B1_A2

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

Time for backsubstitution: 9.34 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4199551, upper bound: 0.4116975
time: 3.57 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4200936, upper bound: 0.4200942
time: 2.85 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -12.8442354, -10.9641628, -12.9913206, -10.9556389, -1.0226493, 1.1910255
1: -10.8961906, -8.8635931, -11.0288706, -8.9068890, -1.2355862, 1.4792786
2: -10.4815531, -8.9550037, -10.5411968, -8.9358397, -1.2222919, 1.1922152
3: -4.1755753, -2.7242639, -4.2976418, -2.7261138, -0.8705430, 1.0408912
4: -14.8858242, -13.0162983, -14.8745136, -12.9899063, -1.0844145, 1.0403459
5: 8.5512571, 9.5715904, 8.5441723, 9.5359926, -0.6614873, 0.6981032
6: -4.3608928, -2.6408846, -4.4150343, -2.6568537, -0.9809954, 1.1005185
7: -15.4902258, -13.4705191, -15.5446548, -13.5213032, -1.2507191, 1.3472834
8: -0.5326333, 0.6971636, -0.4832163, 0.7568560, -0.8208976, 0.7112820
9: -6.4662151, -5.3122053, -6.4618874, -5.2966475, -0.7695677, 0.7089362

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4068354, upper bound: 0.3992621
time: 3.28 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4068354, upper bound: 0.4070038
time: 3.31 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -12.8442402, -10.9583607, -12.9850702, -10.9437981, -1.0173149, 1.2080364
1: -10.8961973, -8.8626394, -11.0082283, -8.9098787, -1.2455778, 1.4651537
2: -10.4815578, -8.9445829, -10.5175018, -8.9254599, -1.2107935, 1.2182589
3: -4.1855903, -2.7242599, -4.3165278, -2.7338285, -0.8919792, 1.0393889
4: -14.8858271, -12.9810448, -14.8852472, -12.9225349, -1.0879250, 1.0873337
5: 8.5355539, 9.5715904, 8.5204678, 9.5298309, -0.6832650, 0.7011580
6: -4.3709798, -2.6408827, -4.4381590, -2.6528771, -0.9972587, 1.1001177
7: -15.4902258, -13.4668713, -15.5361347, -13.5160379, -1.2645302, 1.3435516
8: -0.5450990, 0.6971653, -0.5148637, 0.7654676, -0.8428409, 0.7104535
9: -6.4662151, -5.2924576, -6.4591236, -5.2518821, -0.7544155, 0.7574840

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4068354, upper bound: 0.4047256
time: 3.89 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4068354, upper bound: 0.4124674
time: 2.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.60 seconds
NS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4113316, upper bound: 0.4108485
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4114701, upper bound: 0.4192462
NS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4167950, upper bound: 0.4108485
NS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4169335, upper bound: 0.4192463
NS_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4036753, upper bound: 0.3984131
NS_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4036753, upper bound: 0.4061549
NS_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4036753, upper bound: 0.4038766
NS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4036753, upper bound: 0.4116184
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4144917, upper bound: 0.4116975
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4146302, upper bound: 0.4200951
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4199551, upper bound: 0.4116975
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4200936, upper bound: 0.4200942
NS_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4068354, upper bound: 0.3992621
NS_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4068354, upper bound: 0.4070038
NS_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4068354, upper bound: 0.4047256
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.60
Output dim: 5, lower bound: -0.4068354, upper bound: 0.4124674

## BFS NS instance: NS_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -12.7973938, -11.0090446, -12.8220091, -10.9771080, -0.9606566, 0.9468150
1: -10.9164829, -8.8745279, -10.8900557, -8.8766155, -1.2121677, 1.2276912
2: -10.4899530, -8.9750910, -10.4414749, -8.9609604, -1.1308486, 1.0762663
3: -4.1620402, -2.7233493, -4.1733789, -2.7284780, -0.8370609, 0.8401350
4: -14.8711090, -13.0405931, -14.8843479, -13.0176487, -1.0097647, 0.9876223
5: 8.5695305, 9.5610189, 8.5517588, 9.5547838, -0.6130698, 0.6288394
6: -4.2960033, -2.6882029, -4.3418493, -2.6616247, -0.9180839, 0.9210951
7: -15.4857273, -13.4935703, -15.4852362, -13.4829988, -1.2356138, 1.2355394
8: -0.5066562, 0.6858730, -0.5161443, 0.6961215, -0.6762345, 0.6771994
9: -6.4677582, -5.3448615, -6.4623270, -5.3148804, -0.7076273, 0.6827660

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of NS_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4041231, upper bound: 0.4039788
time: 3.07 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4055272, upper bound: 0.4039788
time: 3.28 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -12.8350258, -10.9785995, -12.8361883, -10.9771080, -0.9586802, 0.9931920
1: -10.9134064, -8.8762627, -10.8915005, -8.8767920, -1.2348852, 1.2118883
2: -10.4907370, -8.9685459, -10.4420433, -8.9591866, -1.1262288, 1.1042373
3: -4.1611433, -2.7278981, -4.1741796, -2.7294974, -0.8372536, 0.8351285
4: -14.8619871, -13.0492554, -14.8814001, -13.0173874, -0.9924936, 1.0052054
5: 8.5624218, 9.5729313, 8.5517902, 9.5591841, -0.6158211, 0.6411611
6: -4.3384018, -2.6509521, -4.3564844, -2.6616292, -0.8898942, 0.9778035
7: -15.4975185, -13.4853344, -15.4870682, -13.4796829, -1.2632461, 1.2487316
8: -0.5065622, 0.6862411, -0.5163338, 0.6963072, -0.6776681, 0.6797748
9: -6.4656124, -5.3447695, -6.4616857, -5.3141890, -0.7061272, 0.6845176

Time for backsubstitution: 8.79 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4056657, upper bound: 0.4148457
time: 3.12 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4056657, upper bound: 0.4123797
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -12.7909546, -10.9972057, -12.8220177, -10.9713058, -0.9777775, 0.9415185
1: -10.8958387, -8.8775196, -10.8900614, -8.8756599, -1.1979203, 1.2377229
2: -10.4662743, -8.9645815, -10.4414768, -8.9505243, -1.1568196, 1.0647407
3: -4.1809816, -2.7311070, -4.1833916, -2.7284729, -0.8355565, 0.8616881
4: -14.8818531, -12.9731894, -14.8843517, -12.9823999, -1.0568643, 0.9911394
5: 8.5458431, 9.5548086, 8.5360708, 9.5547857, -0.6163248, 0.6505009
6: -4.3191938, -2.6842296, -4.3519382, -2.6616251, -0.9177296, 0.9375205
7: -15.4770699, -13.4882908, -15.4852352, -13.4793530, -1.2317247, 1.2496264
8: -0.5383492, 0.6945004, -0.5286088, 0.6961212, -0.6754148, 0.6991756
9: -6.4649944, -5.3000326, -6.4623270, -5.2951331, -0.7561755, 0.6676142

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4167949, upper bound: 0.4038719
time: 3.61 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4167951, upper bound: 0.4038719
time: 3.00 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -12.8286171, -10.9667587, -12.8361979, -10.9713058, -0.9758015, 0.9878983
1: -10.8927650, -8.8792562, -10.8915043, -8.8758373, -1.2206745, 1.2219050
2: -10.4670591, -8.9582367, -10.4420433, -8.9487591, -1.1521597, 1.0927198
3: -4.1800108, -2.7356620, -4.1841941, -2.7294924, -0.8357503, 0.8566544
4: -14.8727274, -12.9818573, -14.8814011, -12.9821358, -1.0395851, 1.0087221
5: 8.5387344, 9.5667915, 8.5361013, 9.5591850, -0.6190028, 0.6629786
6: -4.3615732, -2.6469769, -4.3665724, -2.6616273, -0.8895380, 0.9942367
7: -15.4890585, -13.4800529, -15.4870682, -13.4760399, -1.2595730, 1.2627172
8: -0.5382853, 0.6948695, -0.5288002, 0.6963086, -0.6768483, 0.7017503
9: -6.4628491, -5.2999630, -6.4616857, -5.2944417, -0.7546749, 0.6693649

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4169333, upper bound: 0.4122696
time: 3.01 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4169335, upper bound: 0.4122696
time: 3.21 seconds

## BFS NS instance: NS_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -12.8286285, -10.9624481, -12.9578934, -10.9525480, -0.9681816, 1.1432407
1: -10.8927822, -8.8723869, -11.0002909, -8.9064417, -1.2378092, 1.3788657
2: -10.4670696, -8.9478912, -10.4853401, -8.9707556, -1.1090131, 1.2016754
3: -4.1850719, -2.7356484, -4.3115973, -2.7413278, -0.8846974, 1.0281479
4: -14.8727303, -12.9787655, -14.8833466, -12.9170246, -1.0706491, 1.0781209
5: 8.5359688, 9.5667925, 8.5298958, 9.5248318, -0.6661282, 0.6838417
6: -4.3631201, -2.6469765, -4.4080896, -2.6726496, -0.9352286, 1.0461190
7: -15.4890614, -13.4766579, -15.5283575, -13.5225897, -1.2567492, 1.3304048
8: -0.5394258, 0.6948695, -0.5001340, 0.7445505, -0.8172970, 0.6921952
9: -6.4628491, -5.2939138, -6.4589453, -5.2545557, -0.7295618, 0.7514868

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A1_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3966987, upper bound: 0.4116183
time: 3.48 seconds

## Relational analysis of NS_A1_B1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3966987, upper bound: 0.4046419
time: 3.36 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -12.7980194, -11.0049582, -12.8432198, -10.9683571, -0.9669189, 0.9940662
1: -10.9175234, -8.8821583, -10.8971472, -8.8874636, -1.2107115, 1.2944398
2: -10.5013571, -8.9740963, -10.4727516, -8.9163971, -1.2143102, 1.0901217
3: -4.1624470, -2.7229595, -4.1782713, -2.7254257, -0.8399830, 0.8434230
4: -14.8711338, -13.0432205, -14.8819160, -13.0232830, -1.0121865, 0.9923029
5: 8.5694265, 9.5614824, 8.5424271, 9.5579414, -0.6215956, 0.6428881
6: -4.2962589, -2.6821642, -4.3669209, -2.6418428, -0.9273367, 0.9673212
7: -15.4865742, -13.4940147, -15.4926195, -13.4815197, -1.2387648, 1.2426054
8: -0.5108802, 0.6859040, -0.5305026, 0.7162175, -0.6990991, 0.6938739
9: -6.4667072, -5.3444257, -6.4609795, -5.3125486, -0.7280197, 0.6863062

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.4072832, upper bound: 0.4048278
time: 3.11 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4086873, upper bound: 0.4048277
time: 3.73 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -12.8357582, -10.9745131, -12.8573837, -10.9683571, -0.9648805, 1.0403683
1: -10.9144449, -8.8838940, -10.8985186, -8.8876419, -1.2333665, 1.2785864
2: -10.5021334, -8.9675941, -10.4733353, -8.9148531, -1.2097301, 1.1179299
3: -4.1615443, -2.7275035, -4.1790323, -2.7264357, -0.8401771, 0.8383900
4: -14.8620090, -13.0518894, -14.8789682, -13.0230331, -0.9949164, 1.0098720
5: 8.5623226, 9.5734129, 8.5424604, 9.5623684, -0.6244354, 0.6551708
6: -4.3386598, -2.6449163, -4.3815265, -2.6418457, -0.8991065, 1.0239365
7: -15.4983521, -13.4857731, -15.4944773, -13.4782085, -1.2663441, 1.2557597
8: -0.5108309, 0.6862748, -0.5307426, 0.7164040, -0.7005138, 0.6964077
9: -6.4645624, -5.3443303, -6.4603386, -5.3118715, -0.7264793, 0.6880095

Time for backsubstitution: 8.88 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of NS_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4088258, upper bound: 0.4156947
time: 3.40 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4088258, upper bound: 0.4132288
time: 3.29 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -12.7915754, -10.9931173, -12.8432255, -10.9625549, -0.9840508, 0.9888527
1: -10.8968830, -8.8851471, -10.8971539, -8.8865118, -1.1964703, 1.3044171
2: -10.4776764, -8.9636354, -10.4727535, -8.9059658, -1.2402487, 1.0785522
3: -4.1813660, -2.7307181, -4.1882849, -2.7254210, -0.8384743, 0.8649518
4: -14.8818760, -12.9758205, -14.8819208, -12.9880323, -1.0592728, 0.9958344
5: 8.5457344, 9.5552921, 8.5267487, 9.5579414, -0.6247785, 0.6645681
6: -4.3194466, -2.6781902, -4.3770103, -2.6418407, -0.9270992, 0.9837499
7: -15.4779682, -13.4887304, -15.4926205, -13.4778833, -1.2349114, 1.2565901
8: -0.5425794, 0.6945345, -0.5429668, 0.7162192, -0.6982903, 0.7158294
9: -6.4639430, -5.2996016, -6.4609790, -5.2928023, -0.7765677, 0.6711540

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4199550, upper bound: 0.4047194
time: 5.24 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4199552, upper bound: 0.4047209
time: 3.30 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -12.8293514, -10.9626713, -12.8573895, -10.9625549, -0.9820113, 1.0351577
1: -10.8938007, -8.8868847, -10.8985262, -8.8866873, -1.2191620, 1.2885501
2: -10.4784546, -8.9573345, -10.4733372, -8.9044285, -1.2356305, 1.1063673
3: -4.1803951, -2.7352681, -4.1890469, -2.7264311, -0.8386707, 0.8598924
4: -14.8727531, -12.9844894, -14.8789673, -12.9877834, -1.0419946, 1.0134034
5: 8.5386295, 9.5672894, 8.5267811, 9.5623665, -0.6275499, 0.6770025
6: -4.3618326, -2.6409423, -4.3916168, -2.6418438, -0.8988686, 1.0403678
7: -15.4899387, -13.4804935, -15.4944801, -13.4745750, -1.2627029, 1.2696531
8: -0.5425615, 0.6949034, -0.5432086, 0.7164068, -0.6997054, 0.7183635
9: -6.4617972, -5.2995272, -6.4603386, -5.2921238, -0.7750278, 0.6728566

Time for backsubstitution: 9.32 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4200934, upper bound: 0.4131176
time: 3.09 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4200936, upper bound: 0.4131176
time: 2.99 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -12.8293629, -10.9583607, -12.9792023, -10.9437981, -0.9743814, 1.1905584
1: -10.8938198, -8.8800163, -11.0074463, -8.9172821, -1.2362885, 1.4461622
2: -10.4784660, -8.9469862, -10.5166073, -8.9262352, -1.1928949, 1.2154732
3: -4.1854630, -2.7352552, -4.3164797, -2.7382524, -0.8875327, 1.0315406
4: -14.8727560, -12.9814062, -14.8809109, -12.9226589, -1.0731654, 1.0831850
5: 8.5358629, 9.5672903, 8.5205727, 9.5279922, -0.6747210, 0.6987326
6: -4.3633771, -2.6409407, -4.4335957, -2.6528969, -0.9445181, 1.0926299
7: -15.4899435, -13.4770937, -15.5360031, -13.5210857, -1.2600017, 1.3379755
8: -0.5436933, 0.6949041, -0.5144501, 0.7647390, -0.8402948, 0.7085872
9: -6.4617972, -5.2934775, -6.4575987, -5.2522202, -0.7501030, 0.7549863

Time for backsubstitution: 9.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of NS_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.3998588, upper bound: 0.4124658
time: 2.91 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.3998588, upper bound: 0.4054908
time: 2.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 15.32 seconds
NS_A1_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4041231, upper bound: 0.4039788
NS_A1_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4055272, upper bound: 0.4039788
NS_A1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4056657, upper bound: 0.4148457
NS_A1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4056657, upper bound: 0.4123797
NS_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4167949, upper bound: 0.4038719
NS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4167951, upper bound: 0.4038719
NS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4169333, upper bound: 0.4122696
NS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4169335, upper bound: 0.4122696
NS_A1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.3966987, upper bound: 0.4116183
NS_A1_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.3966987, upper bound: 0.4046419
NS_A1_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4072832, upper bound: 0.4048278
NS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4086873, upper bound: 0.4048277
NS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4088258, upper bound: 0.4156947
NS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4088258, upper bound: 0.4132288
NS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4199550, upper bound: 0.4047194
NS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4199552, upper bound: 0.4047209
NS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4200934, upper bound: 0.4131176
NS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.4200936, upper bound: 0.4131176
NS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.3998588, upper bound: 0.4124658
NS_A1_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 15.32
Output dim: 5, lower bound: -0.3998588, upper bound: 0.4054908

## BFS NS instance: NS_A1_B1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -12.8316307, -10.9796219, -12.8357506, -10.9771080, -0.9526193, 0.9873936
1: -10.8496456, -8.9093142, -10.8705292, -8.8767920, -1.1498880, 1.1046691
2: -10.4775801, -8.9812174, -10.4379559, -8.9597855, -1.1078439, 1.0823178
3: -4.1613221, -2.7422552, -4.1741047, -2.7332768, -0.8127539, 0.8167106
4: -14.8498898, -13.0447092, -14.8784676, -13.0174417, -0.9670413, 0.9828629
5: 8.5738297, 9.5579472, 8.5520382, 9.5568409, -0.5914905, 0.6204008
6: -4.3371925, -2.6514068, -4.3563499, -2.6616774, -0.8877215, 0.9765067
7: -15.4680586, -13.5043745, -15.4810162, -13.4797344, -1.2286906, 1.2108979
8: -0.4751022, 0.6624594, -0.5055337, 0.6962514, -0.6526432, 0.6367199
9: -6.4646277, -5.3472209, -6.4616852, -5.3147788, -0.7045004, 0.6818681

Time for backsubstitution: 9.26 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 3, pos: 2530

## Relational analysis of NS_A1_B1_B1_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4037370, upper bound: 0.4112606
time: 3.06 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4020172, upper bound: 0.4112606
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -12.8347864, -10.9785995, -12.8361149, -10.9771080, -0.9555271, 0.9903901
1: -10.8929110, -8.8762627, -10.8837976, -8.8767920, -1.0979581, 1.2089767
2: -10.4847498, -8.9687023, -10.4390659, -8.9592419, -1.1368375, 1.0867438
3: -4.1611319, -2.7428925, -4.1741748, -2.7347636, -0.8288095, 0.8219442
4: -14.8476496, -13.0492687, -14.8749924, -13.0173931, -0.9675016, 0.9979312
5: 8.5627518, 9.5682430, 8.5518913, 9.5557480, -0.6131831, 0.6190534
6: -4.3382735, -2.6510005, -4.3564415, -2.6616428, -0.8894320, 0.9774220
7: -15.4912100, -13.4853868, -15.4826298, -13.4797020, -1.2281032, 1.2469561
8: -0.5016389, 0.6861696, -0.5148253, 0.6962862, -0.6325030, 0.6781892
9: -6.4656124, -5.3454709, -6.4616857, -5.3143101, -0.7059431, 0.6826499

Time for backsubstitution: 8.66 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 3, pos: 2530

## Relational analysis of NS_A1_B1_B1_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4037370, upper bound: 0.4087902
time: 3.08 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.4020172, upper bound: 0.4087901
time: 3.87 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -12.7909546, -10.9972057, -12.8283558, -10.9874563, -0.9532843, 0.9644439
1: -10.8958387, -8.8775196, -10.9106874, -8.8795376, -1.1945357, 1.2390532
2: -10.4662743, -8.9645815, -10.4651546, -8.9713774, -1.1220427, 1.0984519
3: -4.1809816, -2.7311070, -4.1594248, -2.7207577, -0.8525932, 0.8352709
4: -14.8818531, -12.9731894, -14.8736076, -13.0528746, -0.9916706, 1.0367517
5: 8.5458431, 9.5548086, 8.5624905, 9.5609894, -0.6423473, 0.6156434
6: -4.3191938, -2.6842296, -4.3272524, -2.6656003, -0.9334199, 0.9159949
7: -15.4770699, -13.4882908, -15.4938793, -13.4880257, -1.2230144, 1.2500975
8: -0.5383492, 0.6945004, -0.4957461, 0.6874948, -0.7001067, 0.6658911
9: -6.4649944, -5.3000326, -6.4650908, -5.3459845, -0.6905603, 0.7276275

Time for backsubstitution: 8.90 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.74 + 548.30 = 606.05 seconds
