## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5776130088


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.9959478, -13.6773224, -15.9959478, -13.6773224, -1.6158381, 1.6158381)
1: (-13.0274925, -11.1272545, -13.0274925, -11.1272545, -1.2368627, 1.2368627)
2: (-10.9646196, -9.0909767, -10.9646196, -9.0909767, -1.3015485, 1.3015490)
3: (-14.3623714, -12.5468340, -14.3623714, -12.5468340, -1.3128452, 1.3128452)
4: (7.1985865, 8.7868996, 7.1985865, 8.7868996, -1.4061766, 1.4061770)
5: (-7.0611186, -5.4321260, -7.0611186, -5.4321260, -1.2984648, 1.2984648)
6: (-9.2124720, -6.8170462, -9.2124720, -6.8170462, -1.2679276, 1.2679276)
7: (-5.5251818, -3.9273582, -5.5251818, -3.9273582, -1.2577901, 1.2577901)
8: (-4.1536565, -2.1874776, -4.1536565, -2.1874776, -1.4829001, 1.4829001)
9: (-7.2755570, -5.6725206, -7.2755570, -5.6725206, -1.3555326, 1.3555326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.61 + 37.55 = 61.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.5781902, upper bound: 0.5781905

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6236
type: B, layer: 1, pos: 6236
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6236

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781877
time: 6.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781879, upper bound: 0.5781889
time: 4.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.18 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.18
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781877
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.18
Output dim: 4, lower bound: -0.5781879, upper bound: 0.5781889

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -15.9805241, -13.6872597, -15.9878731, -13.6783161, -1.5984373, 1.5962734
1: -13.0226746, -11.1322098, -13.0257788, -11.1280651, -1.2303219, 1.2283502
2: -10.9416218, -9.1001377, -10.9534054, -9.0917397, -1.2772541, 1.2794089
3: -14.3555412, -12.5518408, -14.3614998, -12.5484638, -1.3019981, 1.3048358
4: 7.2087793, 8.7815847, 7.2031226, 8.7857504, -1.3947773, 1.3914113
5: -7.0478148, -5.4359970, -7.0550809, -5.4323950, -1.2836361, 1.2875323
6: -9.1995001, -6.8427181, -9.2114611, -6.8289137, -1.2427211, 1.2405901
7: -5.5137715, -3.9326763, -5.5205216, -3.9280758, -1.2447300, 1.2469625
8: -4.1263924, -2.2489629, -4.1513176, -2.2163415, -1.4233494, 1.4188704
9: -7.2702656, -5.6781950, -7.2745543, -5.6752510, -1.3474751, 1.3488932

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5763703
time: 4.97 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781889
time: 4.96 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -15.9959469, -13.6773224, -15.9959469, -13.6773224, -1.6023684, 1.6158366
1: -13.0274916, -11.1272564, -13.0274944, -11.1272554, -1.2356215, 1.2402163
2: -10.9646177, -9.0909786, -10.9646187, -9.0909758, -1.2830434, 1.3014917
3: -14.3623724, -12.5468349, -14.3623714, -12.5468330, -1.3108978, 1.3162575
4: 7.1985888, 8.7869015, 7.1985860, 8.7868996, -1.4046679, 1.4039598
5: -7.0611138, -5.4321251, -7.0611157, -5.4321251, -1.3004379, 1.2961707
6: -9.2124710, -6.8170614, -9.2124720, -6.8170500, -1.2679253, 1.2504635
7: -5.5251780, -3.9273586, -5.5251799, -3.9273572, -1.2511549, 1.2577901
8: -4.1536555, -2.1874933, -4.1536527, -2.1874831, -1.4755840, 1.4229517
9: -7.2755575, -5.6725221, -7.2755570, -5.6725230, -1.3555322, 1.3541751

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5763703
time: 5.07 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5781889
time: 5.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.98 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.98
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5763703
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 32.98
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781889
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.98
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5763703
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.98
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5781889

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -15.9805241, -13.6872597, -15.9949055, -13.6773205, -1.5998230, 1.6053162
1: -13.0226746, -11.1322098, -13.0274887, -11.1282711, -1.2287869, 1.2284384
2: -10.9416218, -9.1001377, -10.9634943, -9.0909805, -1.2779469, 1.2912307
3: -14.3555412, -12.5518408, -14.3615704, -12.5475006, -1.3013959, 1.3038526
4: 7.2087793, 8.7815847, 7.1990690, 8.7864513, -1.3932476, 1.3974390
5: -7.0478148, -5.4359970, -7.0611091, -5.4322758, -1.2815685, 1.2918220
6: -9.1995001, -6.8427181, -9.2119331, -6.8170667, -1.2506742, 1.2407231
7: -5.5137715, -3.9326763, -5.5251698, -3.9279146, -1.2448797, 1.2513933
8: -4.1263924, -2.2489629, -4.1531868, -2.1883926, -1.4252253, 1.4132838
9: -7.2702656, -5.6781950, -7.2746367, -5.6725235, -1.3501291, 1.3490171

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781874
time: 5.71 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763672, upper bound: 0.5781856
time: 6.32 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -15.9949055, -13.6773205, -15.9805241, -13.6872597, -1.6053162, 1.5998230
1: -13.0274887, -11.1282711, -13.0226746, -11.1322098, -1.2284384, 1.2287869
2: -10.9634943, -9.0909805, -10.9416218, -9.1001377, -1.2912307, 1.2779469
3: -14.3615704, -12.5475006, -14.3555412, -12.5518408, -1.3038521, 1.3013959
4: 7.1990690, 8.7864513, 7.2087793, 8.7815847, -1.3974395, 1.3932476
5: -7.0611091, -5.4322758, -7.0478148, -5.4359970, -1.2918220, 1.2815685
6: -9.2119331, -6.8170667, -9.1995001, -6.8427181, -1.2407231, 1.2506747
7: -5.5251698, -3.9279146, -5.5137715, -3.9326763, -1.2513933, 1.2448797
8: -4.1531868, -2.1883926, -4.1263924, -2.2489629, -1.4132838, 1.4252253
9: -7.2746367, -5.6725235, -7.2702656, -5.6781950, -1.3490181, 1.3501296

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781863, upper bound: 0.5763205
time: 5.11 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781856, upper bound: 0.5763683
time: 4.94 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -15.9959469, -13.6773224, -15.9959469, -13.6773224, -1.6023664, 1.6023664
1: -13.0274916, -11.1272564, -13.0274916, -11.1272564, -1.2402148, 1.2402148
2: -10.9646177, -9.0909786, -10.9646177, -9.0909786, -1.2830429, 1.2830429
3: -14.3623724, -12.5468349, -14.3623724, -12.5468349, -1.3162565, 1.3162565
4: 7.1985888, 8.7869015, 7.1985888, 8.7869015, -1.4046659, 1.4046655
5: -7.0611138, -5.4321251, -7.0611138, -5.4321251, -1.3004384, 1.3004384
6: -9.2124710, -6.8170614, -9.2124710, -6.8170614, -1.2504640, 1.2504640
7: -5.5251780, -3.9273586, -5.5251780, -3.9273586, -1.2511549, 1.2511549
8: -4.1536555, -2.1874933, -4.1536555, -2.1874933, -1.4229488, 1.4229493
9: -7.2755575, -5.6725221, -7.2755575, -5.6725221, -1.3541751, 1.3541756

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781862, upper bound: 0.5763204
time: 5.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781855, upper bound: 0.5763682
time: 4.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.46 seconds
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 32.46
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781874
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 32.46
Output dim: 4, lower bound: -0.5763672, upper bound: 0.5781856
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.46
Output dim: 4, lower bound: -0.5781863, upper bound: 0.5763205
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.46
Output dim: 4, lower bound: -0.5781856, upper bound: 0.5763683
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.46
Output dim: 4, lower bound: -0.5781862, upper bound: 0.5763204
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.46
Output dim: 4, lower bound: -0.5781855, upper bound: 0.5763682

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -15.9805241, -13.6872597, -15.9942465, -13.6774464, -1.5986872, 1.6035309
1: -13.0226746, -11.1322098, -13.0272560, -11.1315308, -1.2255073, 1.2282176
2: -10.9416218, -9.1001377, -10.9584169, -9.0911598, -1.2777348, 1.2861190
3: -14.3555412, -12.5518408, -14.3606806, -12.5476198, -1.3006210, 1.3023319
4: 7.2087793, 8.7815847, 7.1992874, 8.7861786, -1.3926821, 1.3969064
5: -7.0478148, -5.4359970, -7.0601339, -5.4324074, -1.2814012, 1.2907820
6: -9.1995001, -6.8427181, -9.2111807, -6.8171759, -1.2505503, 1.2400060
7: -5.5137715, -3.9326763, -5.5249381, -3.9284277, -1.2443385, 1.2511044
8: -4.1263924, -2.2489629, -4.1528244, -2.1890304, -1.4240570, 1.4124479
9: -7.2702656, -5.6781950, -7.2744904, -5.6727123, -1.3498945, 1.3488636

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781357
time: 5.41 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781864
time: 6.06 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -15.9805202, -13.6872616, -16.0043354, -13.6639290, -1.6217952, 1.6097412
1: -13.0226727, -11.1322365, -13.0785065, -11.1270504, -1.2364349, 1.2437952
2: -10.9415855, -9.1001387, -10.9689903, -9.0131702, -1.2873907, 1.2981913
3: -14.3555317, -12.5518436, -14.3692274, -12.5336895, -1.3140726, 1.3232489
4: 7.2087803, 8.7815809, 7.1969824, 8.7924776, -1.4042859, 1.3991675
5: -7.0478039, -5.4359980, -7.0655475, -5.4210329, -1.2947593, 1.2958412
6: -9.1994934, -6.8427205, -9.2189550, -6.8052607, -1.2519908, 1.2493205
7: -5.5137687, -3.9326801, -5.5329781, -3.9246941, -1.2489266, 1.2588263
8: -4.1263876, -2.2489634, -4.1667295, -2.1786926, -1.4331546, 1.4180784
9: -7.2702641, -5.6781979, -7.2801352, -5.6678314, -1.3559122, 1.3536811

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5759069, upper bound: 0.5768553
time: 5.05 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5781862
time: 6.67 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.9942465, -13.6774464, -15.9805241, -13.6872597, -1.6035309, 1.5986872
1: -13.0272560, -11.1315308, -13.0226746, -11.1322098, -1.2282176, 1.2255073
2: -10.9584169, -9.0911598, -10.9416218, -9.1001377, -1.2861190, 1.2777343
3: -14.3606806, -12.5476198, -14.3555412, -12.5518408, -1.3023319, 1.3006215
4: 7.1992874, 8.7861786, 7.2087793, 8.7815847, -1.3969064, 1.3926821
5: -7.0601339, -5.4324074, -7.0478148, -5.4359970, -1.2907825, 1.2814016
6: -9.2111807, -6.8171759, -9.1995001, -6.8427181, -1.2400060, 1.2505503
7: -5.5249381, -3.9284277, -5.5137715, -3.9326763, -1.2511044, 1.2443385
8: -4.1528244, -2.1890304, -4.1263924, -2.2489629, -1.4124479, 1.4240570
9: -7.2744904, -5.6727123, -7.2702656, -5.6781950, -1.3488636, 1.3498950

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781344, upper bound: 0.5763205
time: 5.83 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781344, upper bound: 0.5763205
time: 5.09 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16.0043354, -13.6639290, -15.9805202, -13.6872616, -1.6097412, 1.6217952
1: -13.0785065, -11.1270504, -13.0226727, -11.1322365, -1.2437949, 1.2364354
2: -10.9689903, -9.0131702, -10.9415855, -9.1001387, -1.2981915, 1.2873907
3: -14.3692274, -12.5336895, -14.3555317, -12.5518436, -1.3232489, 1.3140721
4: 7.1969824, 8.7924776, 7.2087803, 8.7815809, -1.3991671, 1.4042859
5: -7.0655475, -5.4210329, -7.0478039, -5.4359980, -1.2958407, 1.2947593
6: -9.2189550, -6.8052607, -9.1994934, -6.8427205, -1.2493205, 1.2519908
7: -5.5329781, -3.9246941, -5.5137687, -3.9326801, -1.2588258, 1.2489266
8: -4.1667295, -2.1786926, -4.1263876, -2.2489634, -1.4180784, 1.4331551
9: -7.2801352, -5.6678314, -7.2702641, -5.6781979, -1.3536806, 1.3559117

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5768540, upper bound: 0.5759081
time: 4.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781849, upper bound: 0.5763680
time: 4.85 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.9952917, -13.6774454, -15.9959469, -13.6773224, -1.6005802, 1.6012335
1: -13.0272589, -11.1305161, -13.0274916, -11.1272564, -1.2399945, 1.2369351
2: -10.9595375, -9.0911579, -10.9646177, -9.0909786, -1.2779326, 1.2828298
3: -14.3614807, -12.5469522, -14.3623724, -12.5468349, -1.3147335, 1.3154821
4: 7.1988096, 8.7866259, 7.1985888, 8.7869015, -1.4041309, 1.4040961
5: -7.0601358, -5.4322577, -7.0611138, -5.4321251, -1.2994018, 1.3002706
6: -9.2117205, -6.8171668, -9.2124710, -6.8170614, -1.2497482, 1.2503386
7: -5.5249481, -3.9278717, -5.5251780, -3.9273586, -1.2508664, 1.2506142
8: -4.1532898, -2.1881332, -4.1536555, -2.1874933, -1.4221134, 1.4217825
9: -7.2754097, -5.6727118, -7.2755575, -5.6725221, -1.3540206, 1.3539400

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763205
time: 5.70 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763205
time: 7.55 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.0053768, -13.6639309, -15.9959421, -13.6773214, -1.6067886, 1.6302938
1: -13.0785103, -11.1260357, -13.0274906, -11.1272840, -1.2572756, 1.2478671
2: -10.9701090, -9.0131683, -10.9645796, -9.0909796, -1.2960002, 1.2997470
3: -14.3700304, -12.5330191, -14.3623638, -12.5468349, -1.3356504, 1.3289313
4: 7.1965003, 8.7929268, 7.1985908, 8.7868986, -1.4063268, 1.4156952
5: -7.0655532, -5.4208865, -7.0611062, -5.4321275, -1.3044257, 1.3136187
6: -9.2194920, -6.8052535, -9.2124634, -6.8170609, -1.2590656, 1.2619991
7: -5.5329876, -3.9241405, -5.5251760, -3.9273615, -1.2585869, 1.2552013
8: -4.1671934, -2.1777971, -4.1536536, -2.1875029, -1.4435978, 1.4307518
9: -7.2810545, -5.6678300, -7.2755551, -5.6725230, -1.3588376, 1.3599582

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763671
time: 8.48 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763672
time: 8.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 40.09 seconds
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781357
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5763193, upper bound: 0.5781864
NS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5759069, upper bound: 0.5768553
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5781862
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781344, upper bound: 0.5763205
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781344, upper bound: 0.5763205
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5768540, upper bound: 0.5759081
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781849, upper bound: 0.5763680
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763205
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763205
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763671
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.09
Output dim: 4, lower bound: -0.5781348, upper bound: 0.5763672

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -15.9798660, -13.6873894, -15.9942465, -13.6774464, -1.5969019, 1.6023932
1: -13.0224409, -11.1354694, -13.0272560, -11.1315308, -1.2252874, 1.2249389
2: -10.9365416, -9.1003199, -10.9584169, -9.0911598, -1.2726240, 1.2859058
3: -14.3546524, -12.5519619, -14.3606806, -12.5476198, -1.2991009, 1.3015561
4: 7.2090001, 8.7813072, 7.1992874, 8.7861786, -1.3921471, 1.3963394
5: -7.0468407, -5.4361300, -7.0601339, -5.4324074, -1.2803655, 1.2906146
6: -9.1987495, -6.8428226, -9.2111807, -6.8171759, -1.2498345, 1.2398801
7: -5.5135393, -3.9331899, -5.5249381, -3.9284277, -1.2440491, 1.2505627
8: -4.1260195, -2.2495968, -4.1528244, -2.1890304, -1.4232183, 1.4112811
9: -7.2701197, -5.6783843, -7.2744904, -5.6727123, -1.3497391, 1.3486300

Time for backsubstitution: 22.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 871
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5758493, upper bound: 0.5768002
time: 7.70 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781365
time: 5.00 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -15.9899483, -13.6738586, -15.9942465, -13.6774464, -1.6031370, 1.6161804
1: -13.0737152, -11.1310081, -13.0272560, -11.1315308, -1.2406077, 1.2309008
2: -10.9471388, -9.0223284, -10.9584169, -9.0911598, -1.2831688, 1.2888422
3: -14.3631907, -12.5380220, -14.3606806, -12.5476198, -1.3071680, 1.3150320
4: 7.2067404, 8.7876301, 7.1992874, 8.7861786, -1.3943186, 1.4025617
5: -7.0522208, -5.4247627, -7.0601339, -5.4324074, -1.2853837, 1.3026061
6: -9.2064877, -6.8309021, -9.2111807, -6.8171759, -1.2593803, 1.2473555
7: -5.5216079, -3.9294744, -5.5249381, -3.9284277, -1.2518096, 1.2551084
8: -4.1399727, -2.2392762, -4.1528244, -2.1890304, -1.4275122, 1.4203777
9: -7.2757497, -5.6735015, -7.2744904, -5.6727123, -1.3545666, 1.3537197

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5750121, upper bound: 0.5776933
time: 5.10 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763190, upper bound: 0.5781869
time: 6.85 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -15.9805193, -13.6872644, -16.0043335, -13.6639290, -1.6205330, 1.6094389
1: -13.0224714, -11.1322365, -13.0784225, -11.1270523, -1.2381887, 1.2429078
2: -10.9415836, -9.1001472, -10.9689865, -9.0131750, -1.2855883, 1.2974477
3: -14.3546019, -12.5518475, -14.3688459, -12.5336895, -1.3166838, 1.3227863
4: 7.2087955, 8.7815800, 7.1969872, 8.7924776, -1.4042683, 1.3992233
5: -7.0477967, -5.4359980, -7.0655417, -5.4210334, -1.2965574, 1.2955074
6: -9.1994925, -6.8436546, -9.2189531, -6.8056493, -1.2509298, 1.2468791
7: -5.5136967, -3.9326782, -5.5329475, -3.9246964, -1.2490578, 1.2587843
8: -4.1263847, -2.2489700, -4.1667261, -2.1786966, -1.4311814, 1.4171524
9: -7.2702613, -5.6782022, -7.2801347, -5.6678319, -1.3559098, 1.3517737

Time for backsubstitution: 22.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116
type: A, layer: 1, pos: 116

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781354
time: 6.63 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781862
time: 5.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.9942465, -13.6774464, -15.9798660, -13.6873894, -1.6023932, 1.5969019
1: -13.0272560, -11.1315308, -13.0224409, -11.1354694, -1.2249389, 1.2252874
2: -10.9584169, -9.0911598, -10.9365416, -9.1003199, -1.2859056, 1.2726240
3: -14.3606806, -12.5476198, -14.3546524, -12.5519619, -1.3015561, 1.2991004
4: 7.1992874, 8.7861786, 7.2090001, 8.7813072, -1.3963394, 1.3921471
5: -7.0601339, -5.4324074, -7.0468407, -5.4361300, -1.2906141, 1.2803659
6: -9.2111807, -6.8171759, -9.1987495, -6.8428226, -1.2398801, 1.2498345
7: -5.5249381, -3.9284277, -5.5135393, -3.9331899, -1.2505627, 1.2440491
8: -4.1528244, -2.1890304, -4.1260195, -2.2495968, -1.4112811, 1.4232183
9: -7.2744904, -5.6727123, -7.2701197, -5.6783843, -1.3486295, 1.3497396

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 116
type: B, layer: 1, pos: 116

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 871

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5767986, upper bound: 0.5758506
time: 5.12 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781352, upper bound: 0.5763203
time: 5.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.9942465, -13.6774464, -15.9899483, -13.6738586, -1.6161804, 1.6031375
1: -13.0272560, -11.1315308, -13.0737152, -11.1310081, -1.2309008, 1.2406077
2: -10.9584169, -9.0911598, -10.9471388, -9.0223284, -1.2888422, 1.2831688
3: -14.3606806, -12.5476198, -14.3631907, -12.5380220, -1.3150320, 1.3071680
4: 7.1992874, 8.7861786, 7.2067404, 8.7876301, -1.4025612, 1.3943181
5: -7.0601339, -5.4324074, -7.0522208, -5.4247627, -1.3026056, 1.2853837
6: -9.2111807, -6.8171759, -9.2064877, -6.8309021, -1.2473555, 1.2593803
7: -5.5249381, -3.9284277, -5.5216079, -3.9294744, -1.2551084, 1.2518096
8: -4.1528244, -2.1890304, -4.1399727, -2.2392762, -1.4203777, 1.4275122
9: -7.2744904, -5.6727123, -7.2757497, -5.6735015, -1.3537188, 1.3545666

Time for backsubstitution: 22.32 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 61.16 + 546.22 = 607.38 seconds
