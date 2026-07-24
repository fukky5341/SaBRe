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
execution time: IAR + RelationalAnalysis = 23.76 + 37.73 = 61.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.5781902, upper bound: 0.5781905

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6236
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6236

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781877
time: 7.05 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781879, upper bound: 0.5781889
time: 4.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.15 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.15
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781877
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.15
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

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5763703
time: 5.00 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781889
time: 4.93 seconds

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

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6236
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6236

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5763703
time: 5.02 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5781889
time: 5.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.75 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.75
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5763703
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 32.75
Output dim: 4, lower bound: -0.5763691, upper bound: 0.5781889
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.75
Output dim: 4, lower bound: -0.5781880, upper bound: 0.5763703
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.75
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

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.5759091, upper bound: 0.5768580
time: 4.89 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763689, upper bound: 0.5781886
time: 4.55 seconds

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

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776936, upper bound: 0.5750675
time: 7.34 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781875, upper bound: 0.5763701
time: 6.06 seconds

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

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 871

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776939, upper bound: 0.5750687
time: 4.84 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781877, upper bound: 0.5763701
time: 6.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.69 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.69
Output dim: 4, lower bound: -0.5759091, upper bound: 0.5768580
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 4, lower bound: -0.5763689, upper bound: 0.5781886
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 4, lower bound: -0.5776936, upper bound: 0.5750675
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 4, lower bound: -0.5781875, upper bound: 0.5763701
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 4, lower bound: -0.5776939, upper bound: 0.5750687
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 4, lower bound: -0.5781877, upper bound: 0.5763701

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -15.9805260, -13.6872606, -15.9949026, -13.6773214, -1.6013927, 1.6050172
1: -13.0224733, -11.1322098, -13.0274067, -11.1282701, -1.2305412, 1.2281308
2: -10.9416208, -9.1001463, -10.9634953, -9.0909853, -1.2761803, 1.2912211
3: -14.3546104, -12.5518475, -14.3611841, -12.5475016, -1.3040075, 1.3033900
4: 7.2087955, 8.7815838, 7.1990747, 8.7864494, -1.3932319, 1.3974962
5: -7.0478039, -5.4359961, -7.0611076, -5.4322758, -1.2833662, 1.2914910
6: -9.1994972, -6.8436542, -9.2119341, -6.8174558, -1.2496138, 1.2382026
7: -5.5136995, -3.9326773, -5.5251379, -3.9279151, -1.2450123, 1.2513518
8: -4.1263871, -2.2489686, -4.1531849, -2.1883967, -1.4232469, 1.4123573
9: -7.2702641, -5.6782007, -7.2746358, -5.6725254, -1.3501267, 1.3471098

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781872
time: 6.59 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5781865
time: 5.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.9928913, -13.6781254, -15.9796047, -13.6874285, -1.6044350, 1.6002192
1: -13.0272694, -11.1288023, -13.0225267, -11.1322956, -1.2268152, 1.2282023
2: -10.9612827, -9.0919886, -10.9404888, -9.1002169, -1.2889221, 1.2758021
3: -14.3613043, -12.5461426, -14.3554039, -12.5519037, -1.3030510, 1.3025599
4: 7.1992478, 8.7853680, 7.2088113, 8.7810831, -1.3970094, 1.3920884
5: -7.0597262, -5.4315028, -7.0470524, -5.4360838, -1.2907515, 1.2820539
6: -9.2130527, -6.8172660, -9.1993752, -6.8428102, -1.2386031, 1.2483482
7: -5.5248442, -3.9290376, -5.5135870, -3.9331069, -1.2503114, 1.2434611
8: -4.1498556, -2.1895759, -4.1247911, -2.2490816, -1.4099207, 1.4220433
9: -7.2732258, -5.6746831, -7.2700725, -5.6792932, -1.3464074, 1.3476748

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776333, upper bound: 0.5750672
time: 4.76 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776913, upper bound: 0.5750665
time: 4.79 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.9949036, -13.6773233, -15.9805269, -13.6872597, -1.6068907, 1.5995221
1: -13.0272894, -11.1282721, -13.0225906, -11.1322088, -1.2301936, 1.2284780
2: -10.9634924, -9.0909882, -10.9416218, -9.1001434, -1.2894626, 1.2779369
3: -14.3606396, -12.5475063, -14.3551569, -12.5518436, -1.3064632, 1.3009324
4: 7.1990857, 8.7864494, 7.2087865, 8.7815838, -1.3974218, 1.3933053
5: -7.0611019, -5.4322743, -7.0478106, -5.4359975, -1.2936158, 1.2812400
6: -9.2119350, -6.8180065, -9.1994972, -6.8431044, -1.2404928, 1.2487550
7: -5.5250955, -3.9279137, -5.5137415, -3.9326763, -1.2515264, 1.2448382
8: -4.1531825, -2.1884003, -4.1263890, -2.2489643, -1.4113069, 1.4242983
9: -7.2746348, -5.6725273, -7.2702661, -5.6781998, -1.3490152, 1.3482213

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781342, upper bound: 0.5763687
time: 5.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781849, upper bound: 0.5763671
time: 8.33 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.9939337, -13.6781216, -15.9950237, -13.6774883, -1.6014833, 1.6027637
1: -13.0272732, -11.1277895, -13.0273476, -11.1273394, -1.2385912, 1.2396312
2: -10.9624033, -9.0919876, -10.9634857, -9.0910559, -1.2807279, 1.2808986
3: -14.3621035, -12.5454750, -14.3622332, -12.5468903, -1.3154554, 1.3174210
4: 7.1987677, 8.7858181, 7.1986184, 8.7863998, -1.4042301, 1.4034991
5: -7.0597272, -5.4313540, -7.0602980, -5.4322124, -1.2993765, 1.3009276
6: -9.2135887, -6.8172569, -9.2123518, -6.8171549, -1.2483430, 1.2481532
7: -5.5248528, -3.9284849, -5.5249968, -3.9277897, -1.2500730, 1.2497392
8: -4.1503210, -2.1886802, -4.1520581, -2.1876152, -1.4195776, 1.4202657
9: -7.2741456, -5.6746826, -7.2753625, -5.6736197, -1.3515635, 1.3517199

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776337, upper bound: 0.5750672
time: 4.74 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776915, upper bound: 0.5750649
time: 7.02 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.9959450, -13.6773243, -15.9959450, -13.6773224, -1.6039391, 1.6020660
1: -13.0272923, -11.1272564, -13.0274124, -11.1272564, -1.2419686, 1.2399054
2: -10.9646149, -9.0909872, -10.9646177, -9.0909805, -1.2812760, 1.2830338
3: -14.3614407, -12.5468378, -14.3619881, -12.5468378, -1.3188691, 1.3157949
4: 7.1986036, 8.7869005, 7.1985941, 8.7869015, -1.4046497, 1.4047222
5: -7.0611053, -5.4321251, -7.0611100, -5.4321241, -1.3022308, 1.3001075
6: -9.2124720, -6.8179955, -9.2124691, -6.8174467, -1.2502322, 1.2479434
7: -5.5251064, -3.9273586, -5.5251470, -3.9273586, -1.2512865, 1.2511134
8: -4.1536517, -2.1875019, -4.1536551, -2.1874957, -1.4210443, 1.4229441
9: -7.2755537, -5.6725259, -7.2755561, -5.6725230, -1.3541718, 1.3522649

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 116

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781346, upper bound: 0.5763687
time: 5.76 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781853, upper bound: 0.5763680
time: 8.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 37.25 seconds
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781872
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5781865
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5776333, upper bound: 0.5750672
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5776913, upper bound: 0.5750665
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5781342, upper bound: 0.5763687
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5781849, upper bound: 0.5763671
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5776337, upper bound: 0.5750672
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5776915, upper bound: 0.5750649
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5781346, upper bound: 0.5763687
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.25
Output dim: 4, lower bound: -0.5781853, upper bound: 0.5763680

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.9805260, -13.6872606, -15.9942474, -13.6774464, -1.6002598, 1.6032324
1: -13.0224733, -11.1322098, -13.0271740, -11.1315308, -1.2272611, 1.2279105
2: -10.9416208, -9.1001463, -10.9584150, -9.0911636, -1.2759671, 1.2861094
3: -14.3546104, -12.5518475, -14.3602972, -12.5476208, -1.3032331, 1.3018680
4: 7.2087955, 8.7815838, 7.1992965, 8.7861786, -1.3926668, 1.3969636
5: -7.0478039, -5.4359961, -7.0601268, -5.4324079, -1.2831993, 1.2904506
6: -9.1994972, -6.8436542, -9.2111835, -6.8175597, -1.2494888, 1.2374873
7: -5.5136995, -3.9326773, -5.5249100, -3.9284277, -1.2444706, 1.2510629
8: -4.1263871, -2.2489686, -4.1528254, -2.1890311, -1.4220791, 1.4115224
9: -7.2702641, -5.6782007, -7.2744894, -5.6727157, -1.3498917, 1.3469548

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781354
time: 6.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763191, upper bound: 0.5781865
time: 6.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2

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

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5781339
time: 6.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5763670, upper bound: 0.5781851
time: 6.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.9928913, -13.6781254, -15.9789410, -13.6875572, -1.6032953, 1.5984335
1: -13.0272694, -11.1288023, -13.0222893, -11.1355534, -1.2235365, 1.2279825
2: -10.9612827, -9.0919886, -10.9354115, -9.1003962, -1.2887108, 1.2706919
3: -14.3613043, -12.5461426, -14.3545151, -12.5520191, -1.3022766, 1.3010373
4: 7.1992478, 8.7853680, 7.2090316, 8.7808065, -1.3964410, 1.3915539
5: -7.0597262, -5.4315028, -7.0460801, -5.4362149, -1.2905855, 1.2810178
6: -9.2130527, -6.8172660, -9.1986256, -6.8429189, -1.2384782, 1.2476315
7: -5.5248442, -3.9290376, -5.5133586, -3.9336209, -1.2497697, 1.2431722
8: -4.1498556, -2.1895759, -4.1244192, -2.2497158, -1.4087534, 1.4212065
9: -7.2732258, -5.6746831, -7.2699275, -5.6794829, -1.3461742, 1.3475204

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776333, upper bound: 0.5750121
time: 7.22 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776333, upper bound: 0.5750665
time: 4.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.9928837, -13.6781254, -15.9890203, -13.6740284, -1.6207576, 1.6046643
1: -13.0272694, -11.1288300, -13.0735703, -11.1310921, -1.2344494, 1.2424574
2: -10.9612436, -9.0919924, -10.9460011, -9.0224085, -1.2916546, 1.2887292
3: -14.3612928, -12.5461426, -14.3630552, -12.5380783, -1.3157396, 1.3219643
4: 7.1992483, 8.7853680, 7.2067680, 8.7871284, -1.4080634, 1.3937211
5: -7.0597181, -5.4315033, -7.0513973, -5.4248471, -1.3039389, 1.2860327
6: -9.2130451, -6.8172688, -9.2063713, -6.8309956, -1.2449908, 1.2572150
7: -5.5248413, -3.9290409, -5.5214252, -3.9299083, -1.2543640, 1.2509279
8: -4.1498518, -2.1895809, -4.1383753, -2.2393911, -1.4178524, 1.4267845
9: -7.2732248, -5.6746840, -7.2755580, -5.6745982, -1.3521938, 1.3523316

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776913, upper bound: 0.5750133
time: 4.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5776913, upper bound: 0.5750665
time: 4.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.9949036, -13.6773233, -15.9798660, -13.6873894, -1.6057529, 1.5977373
1: -13.0272894, -11.1282721, -13.0223541, -11.1354694, -1.2269125, 1.2282591
2: -10.9634924, -9.0909882, -10.9365463, -9.1003208, -1.2892509, 1.2728262
3: -14.3606396, -12.5475063, -14.3542671, -12.5519629, -1.3056889, 1.2994118
4: 7.1990857, 8.7864494, 7.2090073, 8.7813091, -1.3968534, 1.3927689
5: -7.0611019, -5.4322743, -7.0468335, -5.4361305, -1.2934494, 1.2802038
6: -9.2119350, -6.8180065, -9.1987486, -6.8432117, -1.2403679, 1.2480392
7: -5.5250955, -3.9279137, -5.5135121, -3.9331894, -1.2509847, 1.2445502
8: -4.1531825, -2.1884003, -4.1260176, -2.2495999, -1.4101405, 1.4234600
9: -7.2746348, -5.6725273, -7.2701187, -5.6783867, -1.3487825, 1.3480654

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 116

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781342, upper bound: 0.5763202
time: 6.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.5781342, upper bound: 0.5763681
time: 6.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.9948978, -13.6773243, -15.9899492, -13.6738596, -1.6215515, 1.6039610
1: -13.0272884, -11.1282969, -13.0736351, -11.1310101, -1.2378287, 1.2430050
2: -10.9634562, -9.0909891, -10.9471378, -9.0223341, -1.2921543, 1.2907050
3: -14.3606329, -12.5475054, -14.3628073, -12.5380201, -1.3191514, 1.3203392
4: 7.1990857, 8.7864485, 7.2067432, 8.7876301, -1.4084797, 1.3949366
5: -7.0610933, -5.4322743, -7.0522151, -5.4247627, -1.3053336, 1.2852187
6: -9.2119246, -6.8180065, -9.2064905, -6.8312874, -1.2470074, 1.2576308
7: -5.5250950, -3.9279184, -5.5215788, -3.9294753, -1.2555804, 1.2523046
8: -4.1531816, -2.1884053, -4.1399717, -2.2392778, -1.4192362, 1.4290442
9: -7.2746353, -5.6725297, -7.2757497, -5.6735034, -1.3548026, 1.3528919

Time for backsubstitution: 22.29 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 61.49 + 548.86 = 610.35 seconds
