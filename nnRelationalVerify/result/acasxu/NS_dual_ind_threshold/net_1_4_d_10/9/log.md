## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 51.042030738


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.1557789, 40.7848358, -22.1557789, 40.7848358, -62.9406128, 62.9406128)
1: (-24.9570370, 38.1305771, -24.9570370, 38.1305771, -63.0876160, 63.0876160)
2: (-25.5280037, 37.2813034, -25.5280037, 37.2813034, -62.8093071, 62.8093071)
3: (-30.7027779, 44.2187729, -30.7027779, 44.2187729, -74.9215546, 74.9215546)
4: (-28.9112663, 41.7009773, -28.9112663, 41.7009773, -70.6122437, 70.6122437)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.71 + 1.54 = 2.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -54.3000327, upper bound: 54.3000327

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2600778, upper bound: 54.2633431
time: 0.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2557767, upper bound: 54.2652887
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.46 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -54.2600778, upper bound: 54.2633431
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -54.2557767, upper bound: 54.2652887

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -18.4868755, 34.6143112, -21.1103840, 38.9932709, -57.4801483, 55.7246895
1: -20.8144035, 32.0120316, -23.7775726, 36.3467484, -57.1611443, 55.7895927
2: -21.3472996, 31.4293518, -24.3371181, 35.5812073, -56.9285011, 55.7664719
3: -25.5772781, 36.8826523, -29.2467651, 42.0971375, -67.6744080, 66.1294098
4: -24.1322937, 34.9039764, -27.5300293, 39.7430649, -63.8753548, 62.4340019

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7808132, upper bound: 54.1953818
time: 0.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7807497, upper bound: 54.1973907
time: 0.51 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -23.0712452, 42.9994736, -21.1892605, 39.1450081, -62.2162552, 64.1887131
1: -25.9661083, 39.6324463, -23.8366375, 36.5415115, -62.5076065, 63.4690742
2: -26.6078491, 38.8119164, -24.4353638, 35.7691040, -62.3769531, 63.2472763
3: -31.9601364, 45.9458580, -29.2917061, 42.3570557, -74.3171921, 75.2375641
4: -30.0656071, 43.3875542, -27.6725006, 39.8568916, -69.9225006, 71.0600586

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2651782, upper bound: 54.2639949
time: 0.74 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2650561, upper bound: 54.2650562
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.00 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -53.7808132, upper bound: 54.1953818
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -53.7807497, upper bound: 54.1973907
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -54.2651782, upper bound: 54.2639949
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -54.2650561, upper bound: 54.2650562

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -17.3100719, 32.6658516, -17.6053886, 33.1633377, -50.4734116, 50.2712402
1: -19.5031128, 30.1052856, -19.8797073, 30.6567421, -50.1598549, 49.9849930
2: -20.0045071, 29.5926323, -20.3359890, 30.1038971, -50.1084023, 49.9286194
3: -23.9766865, 34.6298370, -24.4927387, 35.2814789, -59.2581482, 59.1225662
4: -22.6353683, 32.8023224, -23.0472164, 33.4752502, -56.1106186, 55.8495331

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326395
time: 0.56 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -17.8662910, 33.4648590, -25.1441422, 45.6410408, -63.5073204, 58.6090012
1: -20.1244717, 30.8664398, -28.3127747, 42.6795807, -62.8040428, 59.1792145
2: -20.6360626, 30.3336163, -28.9293518, 41.7163544, -62.3524132, 59.2629700
3: -24.7385235, 35.5491257, -34.8390808, 49.4285278, -74.1670532, 70.3881912
4: -23.3001804, 33.7084503, -32.6048965, 46.8523598, -70.1525192, 66.3133392

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921792
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325316
time: 0.63 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -21.8744087, 40.9598618, -17.6440258, 33.2309570, -55.1053581, 58.6038857
1: -24.6398964, 37.6221275, -19.9072227, 30.6698227, -55.3097191, 57.5293503
2: -25.2417297, 36.8789711, -20.3899269, 30.1120625, -55.3537903, 57.2688980
3: -30.3483238, 43.5675163, -24.5087929, 35.4056244, -65.7539444, 68.0762939
4: -28.5339432, 41.2132721, -23.1363907, 33.4848289, -62.0187683, 64.3496628

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5481539, upper bound: 52.2719610
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2531269, upper bound: 54.2517213
time: 0.65 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -22.4401379, 41.8544235, -25.0359898, 45.4628029, -67.9029388, 66.8904037
1: -25.2644730, 38.5629196, -28.1782665, 42.4772301, -67.7416992, 66.7411880
2: -25.8873367, 37.7805672, -28.8126297, 41.5072632, -67.3945999, 66.5932007
3: -31.1052570, 44.6834373, -34.6624947, 49.3200912, -80.4253464, 79.3459244
4: -29.2486420, 42.2288818, -32.4785271, 46.6203423, -75.8689880, 74.7073975

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7191736, upper bound: 52.6475397
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2530092, upper bound: 54.2530095
time: 0.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.44 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326395
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921792
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325316
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -52.5481539, upper bound: 52.2719610
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -54.2531269, upper bound: 54.2517213
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -52.7191736, upper bound: 52.6475397
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -54.2530092, upper bound: 54.2530095

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.5653133, 26.2878418, -17.1550293, 32.3507767, -45.9160919, 43.4428635
1: -15.3060484, 24.2609882, -19.3733845, 29.9591026, -45.2651520, 43.6343651
2: -15.7388554, 23.9183350, -19.8223190, 29.4270744, -45.1659317, 43.7406502
3: -18.8321304, 27.7950897, -23.8745365, 34.4694595, -53.3015900, 51.6696243
4: -17.8806858, 26.2651653, -22.4885864, 32.6827812, -50.5634651, 48.7537498

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.2901516, 30.9058571, -17.5859108, 33.1296196, -49.4197693, 48.4917679
1: -18.3464165, 28.4012928, -19.8577461, 30.6240501, -48.9704666, 48.2590408
2: -18.8373089, 27.9388561, -20.3136959, 30.0723343, -48.9096413, 48.2525520
3: -22.5406151, 32.6338501, -24.4656715, 35.2429695, -57.7835808, 57.0995216
4: -21.2944794, 30.9155827, -23.0214996, 33.4389076, -54.7333870, 53.9370804

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9326395
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14.0715466, 27.0032005, -24.6870308, 44.7851334, -58.8566818, 51.6902275
1: -15.8724890, 24.9275970, -27.7975216, 41.9534378, -57.8259239, 52.7251205
2: -16.3128586, 24.5781822, -28.4048805, 41.0105438, -57.3234024, 52.9830627
3: -19.5258217, 28.6199818, -34.2120934, 48.5847816, -68.1106033, 62.8320770
4: -18.4776573, 27.0795155, -32.0331726, 46.0312881, -64.5089417, 59.1126747

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7347443, upper bound: 53.6024432
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7347443, upper bound: 53.8921792
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.8113956, 31.6376514, -25.1247940, 45.6041183, -62.4155121, 56.7624435
1: -18.9290314, 29.0926723, -28.2907295, 42.6449852, -61.5740166, 57.3834000
2: -19.4277687, 28.6137619, -28.9070702, 41.6829720, -61.1107407, 57.5208206
3: -23.2557335, 33.4855232, -34.8119354, 49.3881264, -72.6438446, 68.2974396
4: -21.9095230, 31.7485504, -32.5789986, 46.8142281, -68.7237396, 64.3275452

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.9325316
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16.5272484, 31.5238724, -16.2872543, 30.8937073, -47.4209557, 47.8111267
1: -18.5218239, 27.7258568, -18.3710938, 28.2944374, -46.8162613, 46.0969505
2: -19.0564613, 27.3474693, -18.8308182, 27.8297520, -46.8862152, 46.1782684
3: -22.5855637, 31.9918709, -22.5945740, 32.6074562, -55.1930199, 54.5864449
4: -21.2428703, 30.4081039, -21.3370361, 30.8883667, -52.1312256, 51.7451401

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5481539, upper bound: 52.2719610
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5481539, upper bound: 52.2719610
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20.9542160, 39.3412323, -17.6440258, 33.2309570, -54.1851730, 56.9852600
1: -23.6007481, 36.0463486, -19.9072227, 30.6698227, -54.2705688, 55.9535713
2: -24.1861591, 35.3555908, -20.3899269, 30.1120625, -54.2982216, 55.7455177
3: -29.0600777, 41.7227631, -24.5087929, 35.4056244, -64.4656982, 66.2315369
4: -27.3310776, 39.4679909, -23.1363907, 33.4848289, -60.8159065, 62.6043816

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2492236, upper bound: 54.2389037
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2492237, upper bound: 54.2517213
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.7916584, 31.9162617, -23.7209949, 43.0887260, -59.8803864, 55.6372528
1: -18.8257351, 28.1664104, -26.6837234, 40.1119537, -58.9376907, 54.8501205
2: -19.3645248, 27.7672615, -27.3026295, 39.2397499, -58.6042557, 55.0698929
3: -22.9614010, 32.5092850, -32.8017540, 46.5245514, -69.4859543, 65.3110199
4: -21.5822716, 30.8882599, -30.7107849, 44.0267906, -65.6090622, 61.5990448

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7191736, upper bound: 52.6475397
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7189618, upper bound: 52.6461620
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -21.4829102, 40.1669235, -25.0359898, 45.4628029, -66.9457092, 65.2029114
1: -24.1847992, 36.9177017, -28.1782665, 42.4772301, -66.6620331, 65.0959549
2: -24.7893715, 36.1919327, -28.8126297, 41.5072632, -66.2966309, 65.0045624
3: -29.7685909, 42.7547913, -34.6624947, 49.3200912, -79.0886688, 77.4172821
4: -27.9957733, 40.4120827, -32.4785271, 46.6203423, -74.6161118, 72.8905945

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1663141, upper bound: 54.2408352
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1633062, upper bound: 54.1633067
time: 0.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.15 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9326395
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.7347443, upper bound: 53.6024432
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.7347443, upper bound: 53.8921792
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -53.5779449, upper bound: 53.9325316
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -52.5481539, upper bound: 52.2719610
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -52.5481539, upper bound: 52.2719610
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -54.2492236, upper bound: 54.2389037
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -54.2492237, upper bound: 54.2517213
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -52.7191736, upper bound: 52.6475397
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -52.7189618, upper bound: 52.6461620
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -54.1663141, upper bound: 54.2408352
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -54.1633062, upper bound: 54.1633067

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.5653133, 26.2878418, -14.6269293, 28.1050835, -41.6703949, 40.9147720
1: -15.3060484, 24.2609882, -16.5050640, 25.7765064, -41.0825539, 40.7660522
2: -15.7388554, 23.9183350, -16.9442596, 25.4052734, -41.1441193, 40.8625793
3: -18.8321304, 27.7950897, -20.3025112, 29.5556545, -48.3877869, 48.0976028
4: -17.8806858, 26.2651653, -19.2353191, 28.0157719, -45.8964539, 45.5004807

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.5653133, 26.2878418, -19.4682102, 36.7638435, -50.3291550, 45.7560425
1: -15.3060484, 24.2609882, -21.9563541, 33.5662994, -48.8723488, 46.2173386
2: -15.7388554, 23.9183350, -22.4873428, 32.9795151, -48.7183685, 46.4056664
3: -18.8321304, 27.7950897, -27.0707150, 38.7836647, -57.6157951, 54.8658066
4: -17.8806858, 26.2651653, -25.4394817, 36.7927513, -54.6734390, 51.7046471

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.2901516, 30.9058571, -15.0458641, 28.8635597, -45.1537094, 45.9517212
1: -18.3464165, 28.4012928, -16.9748783, 26.4135132, -44.7599297, 45.3761711
2: -18.8373089, 27.9388561, -17.4216194, 26.0274849, -44.8647919, 45.3604736
3: -22.5406151, 32.6338501, -20.8764706, 30.2950630, -52.8356781, 53.5103188
4: -21.2944794, 30.9155827, -19.7484322, 28.7406139, -50.0350952, 50.6640167

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.2901516, 30.9058571, -19.8811607, 37.5255928, -53.8157425, 50.7870178
1: -18.3464165, 28.4012928, -22.4209614, 34.2073975, -52.5538139, 50.8222542
2: -18.8373089, 27.9388561, -22.9597378, 33.6072655, -52.4445724, 50.8985939
3: -22.5406151, 32.6338501, -27.6383095, 39.5313148, -62.0719185, 60.2721481
4: -21.2944794, 30.9155827, -25.9529839, 37.5282135, -58.8226891, 56.8685684

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9326395
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9326395
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.0715466, 27.0032005, -22.4047050, 40.9297066, -55.0012474, 49.4078979
1: -15.8724890, 24.9275970, -25.2192707, 38.1525993, -54.0250893, 50.1468658
2: -16.3128586, 24.5781822, -25.8026314, 37.3606567, -53.6735115, 50.3808022
3: -19.5258217, 28.6199818, -31.0164833, 44.1089745, -63.6347961, 59.6364670
4: -18.4776573, 27.0795155, -29.0994530, 41.7831841, -60.2608376, 56.1789703

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7347443, upper bound: 53.6024432
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7347443, upper bound: 53.6024432
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.0715466, 27.0032005, -26.4922924, 48.6957397, -62.7672768, 53.4954910
1: -15.8724890, 24.9275970, -29.8622818, 44.9779282, -60.8503990, 54.7898788
2: -16.3128586, 24.5781822, -30.5024509, 44.0085297, -60.3213692, 55.0806351
3: -19.5258217, 28.6199818, -36.7986946, 52.1200714, -71.6458893, 65.4186783
4: -18.4776573, 27.0795155, -34.3526382, 49.4721832, -67.9498444, 61.4321518

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7347443, upper bound: 53.8921792
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7347443, upper bound: 53.8921792
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16.8113956, 31.6376514, -22.8416176, 41.7430573, -58.5544357, 54.4792709
1: -18.9290314, 29.0926723, -25.7099228, 38.8408661, -57.7698975, 54.8025932
2: -19.4277687, 28.6137619, -26.3030739, 38.0301208, -57.4578819, 54.9168243
3: -23.2557335, 33.4855232, -31.6140862, 44.9096832, -68.1654205, 65.0996094
4: -21.9095230, 31.7485504, -29.6413403, 42.5626984, -64.4722214, 61.3898926

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.8113956, 31.6376514, -26.9086571, 49.4886360, -66.3000107, 58.5463104
1: -18.9290314, 29.0926723, -30.3303394, 45.6425781, -64.5716095, 59.4230118
2: -19.4277687, 28.6137619, -30.9825535, 44.6568604, -64.0846100, 59.5963135
3: -23.2557335, 33.4855232, -37.3690605, 52.8890076, -76.1447296, 70.8545837
4: -21.9095230, 31.7485504, -34.8712158, 50.2260094, -72.1355286, 66.6197662

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.9325316
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.9325316
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.5272484, 31.5238724, -13.8075294, 26.6832523, -43.2105026, 45.3314018
1: -18.5218239, 27.7258568, -15.5734577, 24.2180195, -42.7398453, 43.2993050
2: -19.0564613, 27.3474693, -16.0061607, 23.9123459, -42.9688072, 43.3536148
3: -22.5855637, 31.9918709, -19.1228905, 27.7277031, -50.3132591, 51.1147537
4: -21.2428703, 30.4081039, -18.0952415, 26.3545341, -47.5974007, 48.5033340

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.5272484, 31.5238724, -18.5770950, 35.2193222, -51.7465706, 50.1009674
1: -18.5218239, 27.7258568, -20.9430294, 31.8451633, -50.3669891, 48.6688843
2: -19.0564613, 27.3474693, -21.4509048, 31.3516502, -50.4081116, 48.7983627
3: -22.5855637, 31.9918709, -25.7859135, 36.7369919, -59.3225517, 57.7777863
4: -21.2428703, 30.4081039, -24.1841850, 34.9533844, -56.1962433, 54.5922890

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -20.9542160, 39.3412323, -15.0645208, 28.8959808, -49.8501968, 54.4057503
1: -23.6007481, 36.0463486, -16.9958324, 26.4444580, -50.0452042, 53.0421829
2: -24.1861591, 35.3555908, -17.4429550, 26.0576000, -50.2437592, 52.7985458
3: -29.0600777, 41.7227631, -20.9023075, 30.3311520, -59.3912277, 62.6250572
4: -27.3310776, 39.4679909, -19.7728100, 28.7748013, -56.1058769, 59.2407990

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2183511, upper bound: 53.9640622
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1449383, upper bound: 53.9621226
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -20.9542160, 39.3412323, -19.8992176, 37.5575714, -58.5117836, 59.2404480
1: -23.6007481, 36.0463486, -22.4412575, 34.2380104, -57.8387604, 58.4876060
2: -24.1861591, 35.3555908, -22.9805012, 33.6369591, -57.8231163, 58.3360863
3: -29.0600777, 41.7227631, -27.6634293, 39.5673676, -68.6274414, 69.3861923
4: -27.3310776, 39.4679909, -25.9768009, 37.5621872, -64.8932571, 65.4447861

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2183513, upper bound: 54.1745228
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1449385, upper bound: 54.1717399
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16.3939075, 31.1647110, -20.0414429, 36.7207794, -53.1146851, 51.2061539
1: -18.3795357, 27.5566540, -22.5604801, 34.3121109, -52.6916466, 50.1171341
2: -18.9108047, 27.1725483, -23.0961170, 33.6238136, -52.5346184, 50.2686653
3: -22.4179554, 31.7989006, -27.7731209, 39.7240753, -62.1420288, 59.5720215
4: -21.0919037, 30.1875420, -26.0417709, 37.5125618, -58.6044655, 56.2293053

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6381671, upper bound: 52.4307821
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3992142, upper bound: 51.8910919
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.7768593, 31.8889160, -22.5298443, 40.8640594, -57.6409187, 54.4187584
1: -18.8091850, 28.1416931, -25.3310871, 37.9992371, -56.8084221, 53.4727745
2: -19.3474712, 27.7431850, -25.9325619, 37.1997986, -56.5472717, 53.6757469
3: -22.9410973, 32.4801483, -31.1308270, 44.0710678, -67.0121536, 63.6109772
4: -21.5626488, 30.8609085, -29.1261482, 41.6985817, -63.2612305, 59.9870567

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.6982642, upper bound: 50.4933043
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.5826065, upper bound: 50.1484970
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -21.0530167, 39.3564987, -21.4256458, 39.1803284, -60.2333374, 60.7821388
1: -23.6999912, 36.2395325, -24.1310387, 36.7772675, -60.4772568, 60.3705597
2: -24.2971573, 35.5314751, -24.6816769, 35.9927368, -60.2898750, 60.2131462
3: -29.1783333, 41.9665108, -29.7331581, 42.6388893, -71.8172150, 71.6996689
4: -27.4606400, 39.6381454, -27.8891182, 40.2276115, -67.6882324, 67.5272675

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1663141, upper bound: 54.2408352
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1663141, upper bound: 54.2404651
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -21.4638329, 40.1314278, -23.8626347, 43.2480164, -64.7118530, 63.9940529
1: -24.1632576, 36.8843307, -26.8428116, 40.3769493, -64.5402069, 63.7271423
2: -24.7674713, 36.1597137, -27.4603443, 39.4827461, -64.2502060, 63.6200485
3: -29.7420406, 42.7157402, -33.0124397, 46.8811111, -76.6231537, 75.7281799
4: -27.9706230, 40.3752365, -30.9084969, 44.3103943, -72.2810135, 71.2837296

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1633062, upper bound: 54.1633067
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1633062, upper bound: 54.1633067
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.92 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9326395
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9326395
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7347443, upper bound: 53.6024432
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7347443, upper bound: 53.6024432
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7347443, upper bound: 53.8921792
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.7347443, upper bound: 53.8921792
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.5779449, upper bound: 53.9325316
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -53.5779449, upper bound: 53.9325316
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.2183511, upper bound: 53.9640622
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.1449383, upper bound: 53.9621226
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.2183513, upper bound: 54.1745228
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.1449385, upper bound: 54.1717399
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 1.92
Output dim: 0, lower bound: -50.6982642, upper bound: 50.4933043
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 1.92
Output dim: 0, lower bound: -50.5826065, upper bound: 50.1484970
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.1663141, upper bound: 54.2408352
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.1663141, upper bound: 54.2404651
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.1633062, upper bound: 54.1633067
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.92
Output dim: 0, lower bound: -54.1633062, upper bound: 54.1633067

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -14.6269293, 28.1050835, -39.5957375, 37.3536530
1: -12.9931536, 20.8490124, -16.5050640, 25.7765064, -38.7696609, 37.3540688
2: -13.3803749, 20.5927849, -16.9442596, 25.4052734, -38.7856407, 37.5370369
3: -16.0014496, 23.8214302, -20.3025112, 29.5556545, -45.5571060, 44.1239395
4: -15.2732220, 22.4852295, -19.2353191, 28.0157719, -43.2889862, 41.7205429

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6240256, upper bound: 53.4654978
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -14.6269293, 28.1050835, -46.4467659, 48.5230370
1: -20.6681404, 31.5833549, -16.5050640, 25.7765064, -46.4446449, 48.0884171
2: -21.1485443, 31.0053368, -16.9442596, 25.4052734, -46.5538177, 47.9495964
3: -25.4536095, 36.4101868, -20.3025112, 29.5556545, -55.0092621, 56.7126961
4: -23.8948879, 34.4529533, -19.2353191, 28.0157719, -51.9106522, 53.6882706

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6240256, upper bound: 53.4654978
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -19.4682102, 36.7638435, -48.2545013, 42.1949310
1: -12.9931536, 20.8490124, -21.9563541, 33.5662994, -46.5594521, 42.8053551
2: -13.3803749, 20.5927849, -22.4873428, 32.9795151, -46.3598862, 43.0801277
3: -16.0014496, 23.8214302, -27.0707150, 38.7836647, -54.7851143, 50.8921432
4: -15.2732220, 22.4852295, -25.4394817, 36.7927513, -52.0659714, 47.9247131

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -19.4682102, 36.7638435, -55.1055260, 53.3643112
1: -20.6681404, 31.5833549, -21.9563541, 33.5662994, -54.2344398, 53.5397034
2: -21.1485443, 31.0053368, -22.4873428, 32.9795151, -54.1280594, 53.4926796
3: -25.4536095, 36.4101868, -27.0707150, 38.7836647, -64.2372742, 63.4809036
4: -23.8948879, 34.4529533, -25.4394817, 36.7927513, -60.6876373, 59.8924332

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -15.0458641, 28.8635597, -42.9758835, 42.2863464
1: -15.9184780, 24.8546124, -16.9748783, 26.4135132, -42.3319931, 41.8294830
2: -16.3577156, 24.5101395, -17.4216194, 26.0274849, -42.3852005, 41.9317589
3: -19.5635452, 28.4796600, -20.8764706, 30.2950630, -49.8586082, 49.3561172
4: -18.5248909, 27.0098228, -19.7484322, 28.7406139, -47.2655029, 46.7582512

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20.3894386, 37.3347435, -15.0458641, 28.8635597, -49.2529984, 52.3806076
1: -22.9784584, 34.5795403, -16.9748783, 26.4135132, -49.3919716, 51.5544205
2: -23.4825687, 33.9048004, -17.4216194, 26.0274849, -49.5100517, 51.3264198
3: -28.2879829, 39.8983345, -20.8764706, 30.2950630, -58.5830421, 60.7748032
4: -26.3920212, 37.9236145, -19.7484322, 28.7406139, -55.1326256, 57.6720467

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -19.8811607, 37.5255928, -51.6379166, 47.1216431
1: -15.9184780, 24.8546124, -22.4209614, 34.2073975, -50.1258698, 47.2755661
2: -16.3577156, 24.5101395, -22.9597378, 33.6072655, -49.9649811, 47.4698792
3: -19.5635452, 28.4796600, -27.6383095, 39.5313148, -59.0948601, 56.1179581
4: -18.5248909, 27.0098228, -25.9529839, 37.5282135, -56.0531044, 52.9627991

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326395
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326393
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.3894386, 37.3347435, -19.8811607, 37.5255928, -57.9150314, 57.2159042
1: -22.9784584, 34.5795403, -22.4209614, 34.2073975, -57.1858444, 57.0004959
2: -23.4825687, 33.9048004, -22.9597378, 33.6072655, -57.0898285, 56.8645363
3: -28.2879829, 39.8983345, -27.6383095, 39.5313148, -67.8192978, 67.5366287
4: -26.3920212, 37.9236145, -25.9529839, 37.5282135, -63.9202271, 63.8765984

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326395
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326393
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -22.4047050, 40.9297066, -52.4203568, 45.1314278
1: -12.9931536, 20.8490124, -25.2192707, 38.1525993, -51.1457520, 46.0682755
2: -13.3803749, 20.5927849, -25.8026314, 37.3606567, -50.7410240, 46.3954048
3: -16.0014496, 23.8214302, -31.0164833, 44.1089745, -60.1104240, 54.8379135
4: -15.2732220, 22.4852295, -29.0994530, 41.7831841, -57.0563927, 51.5846825

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0496457, upper bound: 52.9163575
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6877135, upper bound: 53.5529252
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -22.4047050, 40.9297066, -59.2713890, 56.3008041
1: -20.6681404, 31.5833549, -25.2192707, 38.1525993, -58.8207169, 56.8026199
2: -21.1485443, 31.0053368, -25.8026314, 37.3606567, -58.5092010, 56.8079643
3: -25.4536095, 36.4101868, -31.0164833, 44.1089745, -69.5625839, 67.4266586
4: -23.8948879, 34.4529533, -29.0994530, 41.7831841, -65.6780701, 63.5524025

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0496457, upper bound: 52.9163575
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6877135, upper bound: 53.5529252
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -26.4922924, 48.6957397, -60.1863899, 49.2190170
1: -12.9931536, 20.8490124, -29.8622818, 44.9779282, -57.9710808, 50.7112885
2: -13.3803749, 20.5927849, -30.5024509, 44.0085297, -57.3889046, 51.0952377
3: -16.0014496, 23.8214302, -36.7986946, 52.1200714, -68.1215210, 60.6201248
4: -15.2732220, 22.4852295, -34.3526382, 49.4721832, -64.7454071, 56.8378677

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921790
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921790
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -26.4922924, 48.6957397, -67.0374222, 60.3883896
1: -20.6681404, 31.5833549, -29.8622818, 44.9779282, -65.6460571, 61.4456367
2: -21.1485443, 31.0053368, -30.5024509, 44.0085297, -65.1570740, 61.5077705
3: -25.4536095, 36.4101868, -36.7986946, 52.1200714, -77.5736847, 73.2088776
4: -23.8948879, 34.4529533, -34.3526382, 49.4721832, -73.3670654, 68.8055878

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921790
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921792
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -22.8416176, 41.7430573, -55.8553810, 50.0820999
1: -15.9184780, 24.8546124, -25.7099228, 38.8408661, -54.7593422, 50.5645256
2: -16.3577156, 24.5101395, -26.3030739, 38.0301208, -54.3878288, 50.8132095
3: -19.5635452, 28.4796600, -31.6140862, 44.9096832, -64.4732208, 60.0937386
4: -18.5248909, 27.0098228, -29.6413403, 42.5626984, -61.0875893, 56.6511612

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20.3892612, 37.3345947, -22.8416176, 41.7430573, -62.1322937, 60.1762123
1: -22.9782505, 34.5793762, -25.7099228, 38.8408661, -61.8191147, 60.2892990
2: -23.4823799, 33.9046478, -26.3030739, 38.0301208, -61.5124931, 60.2077141
3: -28.2877445, 39.8981094, -31.6140862, 44.9096832, -73.1974030, 71.5121918
4: -26.3917770, 37.9234200, -29.6413403, 42.5626984, -68.9544754, 67.5647583

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -26.9086571, 49.4886360, -63.6009331, 54.1491394
1: -15.9184780, 24.8546124, -30.3303394, 45.6425781, -61.5610428, 55.1849403
2: -16.3577156, 24.5101395, -30.9825535, 44.6568604, -61.0145645, 55.4926910
3: -19.5635452, 28.4796600, -37.3690605, 52.8890076, -72.4525452, 65.8487244
4: -18.5248909, 27.0098228, -34.8712158, 50.2260094, -68.7509003, 61.8810387

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325316
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325314
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.3892612, 37.3345947, -26.9086571, 49.4886360, -69.8778534, 64.2432556
1: -22.9782505, 34.5793762, -30.3303394, 45.6425781, -68.6208267, 64.9097061
2: -23.4823799, 33.9046478, -30.9825535, 44.6568604, -68.1392288, 64.8871841
3: -28.2877445, 39.8981094, -37.3690605, 52.8890076, -81.1767426, 77.2671661
4: -26.3917770, 37.9234200, -34.8712158, 50.2260094, -76.6177826, 72.7946320

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325316
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325314
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17.5574150, 33.5576134, -14.6269293, 28.1050835, -45.6624985, 48.1845436
1: -19.8183670, 30.7502632, -16.5050640, 25.7765064, -45.5948715, 47.2553177
2: -20.3026810, 30.2188034, -16.9442596, 25.4052734, -45.7079544, 47.1630478
3: -24.4577484, 35.4872437, -20.3025112, 29.5556545, -54.0134048, 55.7897568
4: -23.0012970, 33.5349655, -19.2353191, 28.0157719, -51.0170517, 52.7702866

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2104433, upper bound: 53.9549150
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0974909, upper bound: 53.8838981
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.9114799, 37.4540672, -15.0458641, 28.8635597, -48.7750320, 52.4999313
1: -22.4175568, 34.2604065, -16.9748783, 26.4135132, -48.8310699, 51.2352829
2: -22.9886589, 33.6279831, -17.4216194, 26.0274849, -49.0161400, 51.0496025
3: -27.5933819, 39.6282539, -20.8764706, 30.2950630, -57.8884430, 60.5047150
4: -25.9498520, 37.4774437, -19.7484322, 28.7406139, -54.6904564, 57.2258759

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17.5574150, 33.5576134, -19.4682102, 36.7638435, -54.3212585, 53.0258217
1: -19.8183670, 30.7502632, -21.9563541, 33.5662994, -53.3846664, 52.7066040
2: -20.3026810, 30.2188034, -22.4873428, 32.9795151, -53.2821960, 52.7061386
3: -24.4577484, 35.4872437, -27.0707150, 38.7836647, -63.2414131, 62.5579605
4: -23.0012970, 33.5349655, -25.4394817, 36.7927513, -59.7940483, 58.9744492

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.9114799, 37.4540672, -19.8809242, 37.5253983, -57.4368706, 57.3349915
1: -22.4175568, 34.2604065, -22.4206944, 34.2071800, -56.6247368, 56.6810989
2: -22.9886589, 33.6279831, -22.9594898, 33.6070557, -56.5957108, 56.5874672
3: -27.5933819, 39.6282539, -27.6379910, 39.5310173, -67.1243973, 67.2662430
4: -25.9498520, 37.4774437, -25.9526749, 37.5279579, -63.4777985, 63.4301186

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642078, upper bound: 54.1717398
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18.6135330, 35.2921753, -21.4256458, 39.1803284, -57.7938461, 56.7178192
1: -20.9900990, 32.1215439, -24.1310387, 36.7772675, -57.7673645, 56.2525826
2: -21.5068703, 31.5812416, -24.6816769, 35.9927368, -57.4996071, 56.2629128
3: -25.8637333, 37.0863228, -29.7331581, 42.6388893, -68.5026093, 66.8194809
4: -24.3186264, 35.1811447, -27.8891182, 40.2276115, -64.5462341, 63.0702629

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2408352
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.4063072, 46.8272247, -21.4256458, 39.1803284, -64.5866241, 68.2528687
1: -28.6462059, 43.1931686, -24.1310387, 36.7772675, -65.4234619, 67.3241882
2: -29.2610550, 42.2853508, -24.6816769, 35.9927368, -65.2537918, 66.9670029
3: -35.3114586, 50.0333862, -29.7331581, 42.6388893, -77.9503479, 79.7665405
4: -32.9586563, 47.5135345, -27.8891182, 40.2276115, -73.1862488, 75.4026489

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2404651
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17.9468613, 34.1551132, -23.8626347, 43.2480164, -61.1948776, 58.0177269
1: -20.2491722, 31.4106674, -26.8428116, 40.3769493, -60.6261101, 58.2534676
2: -20.7486954, 30.8505535, -27.4603443, 39.4827461, -60.2314339, 58.3108978
3: -24.9880733, 36.2773743, -33.0124397, 46.8811111, -71.8691864, 69.2898102
4: -23.5056725, 34.2449112, -30.9084969, 44.3103943, -67.8160629, 65.1533966

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.4113770, 38.2129784, -23.8626347, 43.2480164, -63.6593742, 62.0755997
1: -22.9685669, 35.0723991, -26.8428116, 40.3769493, -63.3455162, 61.9152031
2: -23.5585613, 34.4084091, -27.4603443, 39.4827461, -63.0412941, 61.8687515
3: -28.2634182, 40.5897751, -33.0124397, 46.8811111, -75.1445312, 73.6022186
4: -26.5753155, 38.3587837, -30.9084969, 44.3103943, -70.8856964, 69.2672729

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067
time: 0.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.05 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7586065, upper bound: 53.9004552
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7564414, upper bound: 53.8986687
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6105318, upper bound: 53.9101079
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326395
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326393
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326395
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6129414, upper bound: 53.9326393
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.0496457, upper bound: 52.9163575
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6877135, upper bound: 53.5529252
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.0496457, upper bound: 52.9163575
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6877135, upper bound: 53.5529252
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921790
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921790
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921790
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.7546576, upper bound: 53.8921792
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.5779449, upper bound: 53.5779449
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325316
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325314
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325316
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.6116097, upper bound: 53.9325314
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.2104433, upper bound: 53.9549150
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.0974909, upper bound: 53.8838981
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -54.1642078, upper bound: 54.1717398
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2408352
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2404651
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.05
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -11.4906597, 22.7267227, -34.2173805, 34.2173805
1: -12.9931536, 20.8490124, -12.9931536, 20.8490124, -33.8421669, 33.8421669
2: -13.3803749, 20.5927849, -13.3803749, 20.5927849, -33.9731598, 33.9731598
3: -16.0014496, 23.8214302, -16.0014496, 23.8214302, -39.8228798, 39.8228798
4: -15.2732220, 22.4852295, -15.2732220, 22.4852295, -37.7584534, 37.7584534

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2238161, upper bound: 54.1533464
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2368939, upper bound: 54.2034040
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -14.1123228, 27.2404823, -38.7311363, 36.8390465
1: -12.9931536, 20.8490124, -15.9184780, 24.8546124, -37.8477592, 36.7674828
2: -13.3803749, 20.5927849, -16.3577156, 24.5101395, -37.8905144, 36.9504967
3: -16.0014496, 23.8214302, -19.5635452, 28.4796600, -44.4811020, 43.3849754
4: -15.2732220, 22.4852295, -18.5248909, 27.0098228, -42.2830429, 41.0101204

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2238161, upper bound: 54.1533464
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2368939, upper bound: 54.2034040
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -11.4906597, 22.7267227, -41.0684052, 45.3867607
1: -20.6681404, 31.5833549, -12.9931536, 20.8490124, -41.5171471, 44.5765076
2: -21.1485443, 31.0053368, -13.3803749, 20.5927849, -41.7413292, 44.3857117
3: -25.4536095, 36.4101868, -16.0014496, 23.8214302, -49.2750359, 52.4116364
4: -23.8948879, 34.4529533, -15.2732220, 22.4852295, -46.3801193, 49.7261734

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7137535, upper bound: 53.5314479
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5732737, upper bound: 53.3574252
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -14.1123228, 27.2404823, -45.5821648, 48.0084305
1: -20.6681404, 31.5833549, -15.9184780, 24.8546124, -45.5227394, 47.5018311
2: -21.1485443, 31.0053368, -16.3577156, 24.5101395, -45.6586838, 47.3630524
3: -25.4536095, 36.4101868, -19.5635452, 28.4796600, -53.9332504, 55.9737320
4: -23.8948879, 34.4529533, -18.5248909, 27.0098228, -50.9047050, 52.9778442

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7137535, upper bound: 53.5314479
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5732737, upper bound: 53.3574252
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -16.5780373, 31.9165230, -43.4071655, 39.3047600
1: -12.9931536, 20.8490124, -18.7406483, 29.0654297, -42.0585823, 39.5896568
2: -13.3803749, 20.5927849, -19.1804962, 28.6155891, -41.9959641, 39.7732773
3: -16.0014496, 23.8214302, -23.1438141, 33.4699211, -49.4713707, 46.9652443
4: -15.2732220, 22.4852295, -21.7336960, 31.7478294, -47.0210457, 44.2189255

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2186915, upper bound: 54.1707650
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2265096, upper bound: 54.1726818
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -18.8830833, 35.7308578, -47.2215118, 41.6098061
1: -12.9931536, 20.8490124, -21.2896595, 32.5061417, -45.4992943, 42.1386642
2: -13.3803749, 20.5927849, -21.8136520, 31.9565125, -45.3368835, 42.4064369
3: -16.0014496, 23.8214302, -26.2257919, 37.5368881, -53.5383377, 50.0472221
4: -15.2732220, 22.4852295, -24.6326466, 35.6247749, -50.8979950, 47.1178741

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2186915, upper bound: 54.1707650
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2265096, upper bound: 54.1726818
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -16.5780373, 31.9165230, -50.2582054, 50.4741440
1: -20.6681404, 31.5833549, -18.7406483, 29.0654297, -49.7335701, 50.3240013
2: -21.1485443, 31.0053368, -19.1804962, 28.6155891, -49.7641335, 50.1858330
3: -25.4536095, 36.4101868, -23.1438141, 33.4699211, -58.9235306, 59.5540009
4: -23.8948879, 34.4529533, -21.7336960, 31.7478294, -55.6427155, 56.1866493

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4828428, upper bound: 53.7898940
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5259819
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7508023, upper bound: 53.8933997
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -18.8830833, 35.7308578, -54.0725403, 52.7791901
1: -20.6681404, 31.5833549, -21.2896595, 32.5061417, -53.1742821, 52.8730164
2: -21.1485443, 31.0053368, -21.8136520, 31.9565125, -53.1050568, 52.8189888
3: -25.4536095, 36.4101868, -26.2257919, 37.5368881, -62.9904938, 62.6359787
4: -23.8948879, 34.4529533, -24.6326466, 35.6247749, -59.5196571, 59.0855980

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4828428, upper bound: 53.7898940
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5259819
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7508023, upper bound: 53.8933997
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -11.4906597, 22.7267227, -36.8390465, 38.7311401
1: -15.9184780, 24.8546124, -12.9931536, 20.8490124, -36.7674828, 37.8477592
2: -16.3577156, 24.5101395, -13.3803749, 20.5927849, -36.9505005, 37.8905144
3: -19.5635452, 28.4796600, -16.0014496, 23.8214302, -43.3849754, 44.4811020
4: -18.5248909, 27.0098228, -15.2732220, 22.4852295, -41.0101204, 42.2830429

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -14.1123228, 27.2404823, -41.3528061, 41.3528061
1: -15.9184780, 24.8546124, -15.9184780, 24.8546124, -40.7730789, 40.7730827
2: -16.3577156, 24.5101395, -16.3577156, 24.5101395, -40.8678551, 40.8678551
3: -19.5635452, 28.4796600, -19.5635452, 28.4796600, -48.0432053, 48.0432053
4: -18.5248909, 27.0098228, -18.5248909, 27.0098228, -45.5347099, 45.5347099

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -20.3894386, 37.3347435, -11.4906597, 22.7267227, -43.1161613, 48.8253975
1: -22.9784584, 34.5795403, -12.9931536, 20.8490124, -43.8274651, 47.5726929
2: -23.4825687, 33.9048004, -13.3803749, 20.5927849, -44.0753479, 47.2851715
3: -28.2879829, 39.8983345, -16.0014496, 23.8214302, -52.1094131, 55.8997841
4: -26.3920212, 37.9236145, -15.2732220, 22.4852295, -48.8772469, 53.1968346

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4529032, upper bound: 53.8119067
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9204798, upper bound: 53.0603758
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5634923, upper bound: 53.8864395
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -20.3894386, 37.3347435, -14.1123228, 27.2404823, -47.6299210, 51.4470673
1: -22.9784584, 34.5795403, -15.9184780, 24.8546124, -47.8330574, 50.4980164
2: -23.4825687, 33.9048004, -16.3577156, 24.5101395, -47.9927063, 50.2625122
3: -28.2879829, 39.8983345, -19.5635452, 28.4796600, -56.7676315, 59.4618797
4: -26.3920212, 37.9236145, -18.5248909, 27.0098228, -53.4018364, 56.4485054

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4529032, upper bound: 53.8119067
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9204798, upper bound: 53.0603758
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5634923, upper bound: 53.8864395
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -16.5780373, 31.9165230, -46.0288467, 43.8185196
1: -15.9184780, 24.8546124, -18.7406483, 29.0654297, -44.9839058, 43.5952568
2: -16.3577156, 24.5101395, -19.1804962, 28.6155891, -44.9733047, 43.6906357
3: -19.5635452, 28.4796600, -23.1438141, 33.4699211, -53.0334663, 51.6234703
4: -18.5248909, 27.0098228, -21.7336960, 31.7478294, -50.2727165, 48.7435150

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1110167, upper bound: 54.1683255
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1902880, upper bound: 54.1716890
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -18.8832836, 35.7310181, -49.8433418, 46.1237640
1: -15.9184780, 24.8546124, -21.2898827, 32.5063171, -48.4247971, 46.1444893
2: -16.3577156, 24.5101395, -21.8138657, 31.9566841, -48.3143997, 46.3240051
3: -19.5635452, 28.4796600, -26.2260628, 37.5371437, -57.1006889, 54.7057114
4: -18.5248909, 27.0098228, -24.6329060, 35.6249886, -54.1498795, 51.6427307

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1110167, upper bound: 54.1683255
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1902880, upper bound: 54.1716890
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -20.3894386, 37.3347435, -16.5780373, 31.9165230, -52.3059616, 53.9127808
1: -22.9784584, 34.5795403, -18.7406483, 29.0654297, -52.0438881, 53.3201904
2: -23.4825687, 33.9048004, -19.1804962, 28.6155891, -52.0981560, 53.0852890
3: -28.2879829, 39.8983345, -23.1438141, 33.4699211, -61.7579041, 63.0421486
4: -26.3920212, 37.9236145, -21.7336960, 31.7478294, -58.1398354, 59.6573105

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4857279, upper bound: 53.8548590
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6073332, upper bound: 53.9251114
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5432133, upper bound: 53.9087205
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -20.3894386, 37.3347435, -18.8832836, 35.7310181, -56.1204567, 56.2180214
1: -22.9784584, 34.5795403, -21.2898827, 32.5063171, -55.4847755, 55.8694229
2: -23.4825687, 33.9048004, -21.8138657, 31.9566841, -55.4392395, 55.7186623
3: -28.2879829, 39.8983345, -26.2260628, 37.5371437, -65.8251114, 66.1243973
4: -26.3920212, 37.9236145, -24.6329060, 35.6249886, -62.0169907, 62.5565186

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4857279, upper bound: 53.8548592
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6073332, upper bound: 53.9251113
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5432133, upper bound: 53.9087205
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.2119761, 20.3488426, -17.1894398, 31.7353821, -41.9473572, 37.5382843
1: -11.5452738, 18.5013504, -19.2547836, 28.6063499, -40.1516228, 37.7561340
2: -11.9215336, 18.3030338, -19.7952690, 28.1890373, -40.1105728, 38.0983047
3: -14.1903076, 21.0924492, -23.4754105, 32.9487686, -47.1390762, 44.5678596
4: -13.5824699, 19.9194107, -22.0366192, 31.3183403, -44.9008102, 41.9560318

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7583243, upper bound: 52.7925034
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0515877, upper bound: 52.9141563
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -21.4790726, 39.2890244, -50.7796783, 44.2057953
1: -12.9931536, 20.8490124, -24.1658669, 36.5254707, -49.5186234, 45.0148773
2: -13.3803749, 20.5927849, -24.7398090, 35.7934265, -49.1738014, 45.3325882
3: -16.0014496, 23.8214302, -29.7014160, 42.2039642, -58.2054100, 53.5228462
4: -15.2732220, 22.4852295, -27.8689404, 39.9814491, -55.2546692, 50.3541679

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7650425, upper bound: 53.3960800
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9750785, upper bound: 53.5673683
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.0752907, 31.6702652, -17.1894398, 31.7353821, -48.8106728, 48.8597031
1: -19.2278805, 29.3389397, -19.2547836, 28.6063499, -47.8342247, 48.5937195
2: -19.7006016, 28.8517761, -19.7952690, 28.1890373, -47.8896408, 48.6470451
3: -23.6499920, 33.7713776, -23.4754105, 32.9487686, -56.5987587, 57.2467842
4: -22.2003593, 32.0098038, -22.0366192, 31.3183403, -53.5186996, 54.0464249

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.4628825, upper bound: 52.6763389
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3065274, upper bound: 52.6195791
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -21.4790726, 39.2890244, -57.6307068, 55.3751793
1: -20.6681404, 31.5833549, -24.1658669, 36.5254707, -57.1936073, 55.7492218
2: -21.1485443, 31.0053368, -24.7398090, 35.7934265, -56.9419708, 55.7451439
3: -25.4536095, 36.4101868, -29.7014160, 42.2039642, -67.6575775, 66.1116028
4: -23.8948879, 34.4529533, -27.8689404, 39.9814491, -63.8763313, 62.3218918

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8539414, upper bound: 52.8583024
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6903222, upper bound: 52.8214052
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -23.5083370, 43.5609932, -55.0516510, 46.2350616
1: -12.9931536, 20.8490124, -26.5329418, 40.2712250, -53.2643776, 47.3819504
2: -13.3803749, 20.5927849, -27.0840607, 39.4476700, -52.8280411, 47.6768456
3: -16.0014496, 23.8214302, -32.7493973, 46.5774879, -62.5789299, 56.5708275
4: -15.2732220, 22.4852295, -30.5417747, 44.2039719, -59.4771957, 53.0270004

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8253629, upper bound: 53.6542159
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2179532, upper bound: 54.1085246
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.4906597, 22.7267227, -25.8506546, 47.4542923, -58.9449539, 48.5773773
1: -12.9931536, 20.8490124, -29.1244144, 43.7585106, -56.7516632, 49.9734154
2: -13.3803749, 20.5927849, -29.7637978, 42.8362465, -56.2166214, 50.3565826
3: -16.0014496, 23.8214302, -35.8765335, 50.7101250, -66.7115707, 59.6979637
4: -15.2732220, 22.4852295, -33.4723434, 48.1443787, -63.4175949, 55.9575729

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8253629, upper bound: 53.6542159
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2179532, upper bound: 54.1085246
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -23.5083370, 43.5609932, -61.9026756, 57.4044418
1: -20.6681404, 31.5833549, -26.5329418, 40.2712250, -60.9393654, 58.1162949
2: -21.1485443, 31.0053368, -27.0840607, 39.4476700, -60.5962105, 58.0893974
3: -25.4536095, 36.4101868, -32.7493973, 46.5774879, -72.0310974, 69.1595840
4: -23.8948879, 34.4529533, -30.5417747, 44.2039719, -68.0988541, 64.9947281

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5168676
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7490094, upper bound: 53.8883722
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -18.3416824, 33.8961067, -25.8506546, 47.4542923, -65.7959747, 59.7467575
1: -20.6681404, 31.5833549, -29.1244144, 43.7585106, -64.4266510, 60.7077713
2: -21.1485443, 31.0053368, -29.7637978, 42.8362465, -63.9847908, 60.7691269
3: -25.4536095, 36.4101868, -35.8765335, 50.7101250, -76.1637344, 72.2867203
4: -23.8948879, 34.4529533, -33.4723434, 48.1443787, -72.0392609, 67.9252930

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5168676
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7490094, upper bound: 53.8883724
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -19.0125427, 35.1322556, -49.2445793, 46.2530251
1: -15.9184780, 24.8546124, -21.4134941, 32.7879028, -48.7063828, 46.2680855
2: -16.3577156, 24.5101395, -21.9236660, 32.1679916, -48.5257034, 46.4338074
3: -19.5635452, 28.4796600, -26.3699760, 37.8068047, -57.3703499, 54.8496246
4: -18.5248909, 27.0098228, -24.7712917, 35.7631111, -54.2880020, 51.7811127

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9409143, upper bound: 53.0935559
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8759553, upper bound: 53.5634925
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -21.6481285, 39.5172691, -53.6295929, 48.8886108
1: -15.9184780, 24.8546124, -24.3567982, 36.7261505, -52.6446266, 49.2114029
2: -16.3577156, 24.5101395, -24.9315796, 35.9881516, -52.3458672, 49.4417191
3: -19.5635452, 28.4796600, -29.9461422, 42.4385223, -62.0020638, 58.4257965
4: -18.5248909, 27.0098228, -28.0547447, 40.2319031, -58.7567863, 55.0645638

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9409143, upper bound: 53.0935559
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8759553, upper bound: 53.5634925
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -20.3892612, 37.3345947, -19.0125427, 35.1322556, -55.5215111, 56.3471375
1: -22.9782505, 34.5793762, -21.4134941, 32.7879028, -55.7661514, 55.9928627
2: -23.4823799, 33.9046478, -21.9236660, 32.1679916, -55.6503716, 55.8283081
3: -28.2877445, 39.8981094, -26.3699760, 37.8068047, -66.0945511, 66.2680817
4: -26.3917770, 37.9234200, -24.7712917, 35.7631111, -62.1548767, 62.6947098

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9153733, upper bound: 53.0435062
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5258719, upper bound: 53.5258721
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -20.3892612, 37.3345947, -21.6481285, 39.5172691, -59.9065170, 58.9827232
1: -22.9782505, 34.5793762, -24.3567982, 36.7261505, -59.7043991, 58.9361649
2: -23.4823799, 33.9046478, -24.9315796, 35.9881516, -59.4705238, 58.8362160
3: -28.2877445, 39.8981094, -29.9461422, 42.4385223, -70.7262573, 69.8442535
4: -26.3917770, 37.9234200, -28.0547447, 40.2319031, -66.6236649, 65.9781647

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9153733, upper bound: 53.0435062
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5258719, upper bound: 53.5258721
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -23.5083370, 43.5609932, -57.6733170, 50.7488174
1: -15.9184780, 24.8546124, -26.5329418, 40.2712250, -56.1897049, 51.3875427
2: -16.3577156, 24.5101395, -27.0840607, 39.4476700, -55.8053780, 51.5942001
3: -19.5635452, 28.4796600, -32.7493973, 46.5774879, -66.1410217, 61.2290497
4: -18.5248909, 27.0098228, -30.5417747, 44.2039719, -62.7288551, 57.5515900

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9698747, upper bound: 54.1400144
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8979122, upper bound: 53.9470216
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.1123228, 27.2404823, -25.8506546, 47.4542923, -61.5666161, 53.0911331
1: -15.9184780, 24.8546124, -29.1244144, 43.7585106, -59.6769867, 53.9790115
2: -16.3577156, 24.5101395, -29.7637978, 42.8362465, -59.1939621, 54.2739372
3: -19.5635452, 28.4796600, -35.8765335, 50.7101250, -70.2736588, 64.3561935
4: -18.5248909, 27.0098228, -33.4723434, 48.1443787, -66.6692657, 60.4821663

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9698747, upper bound: 54.1400144
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8979122, upper bound: 53.9470218
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -20.3892612, 37.3345947, -23.5083370, 43.5609932, -63.9502296, 60.8429337
1: -22.9782505, 34.5793762, -26.5329418, 40.2712250, -63.2494736, 61.1123199
2: -23.4823799, 33.9046478, -27.0840607, 39.4476700, -62.9300385, 60.9887009
3: -28.2877445, 39.8981094, -32.7493973, 46.5774879, -74.8652191, 72.6475067
4: -26.3917770, 37.9234200, -30.5417747, 44.2039719, -70.5957413, 68.4651947

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4799647, upper bound: 53.8514827
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5544349, upper bound: 53.6848892
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5139289, upper bound: 53.6943285
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -20.3892612, 37.3345947, -25.8506546, 47.4542923, -67.8435364, 63.1852493
1: -22.9782505, 34.5793762, -29.1244144, 43.7585106, -66.7367630, 63.7037849
2: -23.4823799, 33.9046478, -29.7637978, 42.8362465, -66.3186264, 63.6684380
3: -28.2877445, 39.8981094, -35.8765335, 50.7101250, -78.9978561, 75.7746429
4: -26.3917770, 37.9234200, -33.4723434, 48.1443787, -74.5361481, 71.3957672

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4799647, upper bound: 53.8514825
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5544349, upper bound: 53.6848892
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5139290, upper bound: 53.6943283
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17.5574150, 33.5576134, -13.1639166, 25.6805420, -43.2379570, 46.7215309
1: -19.8183670, 30.7502632, -14.8816490, 23.4039650, -43.2223244, 45.6319046
2: -20.3026810, 30.2188034, -15.2884903, 23.1040649, -43.4067459, 45.5072861
3: -24.4577484, 35.4872437, -18.3029613, 26.7559109, -51.2136574, 53.7902069
4: -23.0012970, 33.5349655, -17.3561916, 25.4129257, -48.4142227, 50.8911591

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2104433, upper bound: 53.9549150
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5837629, upper bound: 53.4268794
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.9095211, 32.4723282, -14.1127777, 27.2896061, -44.1991272, 46.5851059
1: -19.0890884, 29.7002850, -15.9271898, 25.0356464, -44.1247292, 45.6274757
2: -19.5626850, 29.2018681, -16.3770714, 24.6659260, -44.2285919, 45.5789299
3: -23.5487213, 34.2473145, -19.5351105, 28.6631088, -52.2118301, 53.7824135
4: -22.1712494, 32.3617821, -18.6446342, 27.1196365, -49.2908859, 51.0064087

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0974909, upper bound: 53.8838981
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0974909, upper bound: 53.8838981
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19.9114799, 37.4540672, -11.4906597, 22.7267227, -42.6382027, 48.9447212
1: -22.4175568, 34.2604065, -12.9931536, 20.8490124, -43.2665596, 47.2535591
2: -22.9886589, 33.6279831, -13.3803749, 20.5927849, -43.5814438, 47.0083542
3: -27.5933819, 39.6282539, -16.0014496, 23.8214302, -51.4148102, 55.6296997
4: -25.9498520, 37.4774437, -15.2732220, 22.4852295, -48.4350777, 52.7506638

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19.9114799, 37.4540672, -14.1123228, 27.2404823, -47.1519585, 51.5663910
1: -22.4175568, 34.2604065, -15.9184780, 24.8546124, -47.2721634, 50.1788864
2: -22.9886589, 33.6279831, -16.3577156, 24.5101395, -47.4987984, 49.9856987
3: -27.5933819, 39.6282539, -19.5635452, 28.4796600, -56.0730286, 59.1917877
4: -25.9498520, 37.4774437, -18.5248909, 27.0098228, -52.9596634, 56.0023346

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17.5574150, 33.5576134, -16.5780373, 31.9165230, -49.4739380, 50.1356506
1: -19.8183670, 30.7502632, -18.7406483, 29.0654297, -48.8837929, 49.4909019
2: -20.3026810, 30.2188034, -19.1804962, 28.6155891, -48.9182701, 49.3992920
3: -24.4577484, 35.4872437, -23.1438141, 33.4699211, -57.9276695, 58.6310577
4: -23.0012970, 33.5349655, -21.7336960, 31.7478294, -54.7491188, 55.2686615

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2183512, upper bound: 54.1745227
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2183512, upper bound: 54.1745227
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17.5574150, 33.5576134, -18.8830833, 35.7308578, -53.2882729, 52.4406967
1: -19.8183670, 30.7502632, -21.2896595, 32.5061417, -52.3245087, 52.0399132
2: -20.3026810, 30.2188034, -21.8136520, 31.9565125, -52.2591934, 52.0324554
3: -24.4577484, 35.4872437, -26.2257919, 37.5368881, -61.9946213, 61.7130318
4: -23.0012970, 33.5349655, -24.6326466, 35.6247749, -58.6260719, 58.1676102

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2183512, upper bound: 54.1745227
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2412284, upper bound: 54.1745227
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -19.9114799, 37.4540672, -16.5780373, 31.9165230, -51.8279877, 54.0321045
1: -22.4175568, 34.2604065, -18.7406483, 29.0654297, -51.4829865, 53.0010529
2: -22.9886589, 33.6279831, -19.1804962, 28.6155891, -51.6042480, 52.8084755
3: -27.5933819, 39.6282539, -23.1438141, 33.4699211, -61.0633011, 62.7720680
4: -25.9498520, 37.4774437, -21.7336960, 31.7478294, -57.6976776, 59.2111397

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -19.9114799, 37.4540672, -18.8830833, 35.7308578, -55.6423302, 56.3371506
1: -22.4175568, 34.2604065, -21.2896595, 32.5061417, -54.9236984, 55.5500641
2: -22.9886589, 33.6279831, -21.8136520, 31.9565125, -54.9451714, 55.4416351
3: -27.5933819, 39.6282539, -26.2257919, 37.5368881, -65.1302490, 65.8540497
4: -25.9498520, 37.4774437, -24.6326466, 35.6247749, -61.5746193, 62.1100922

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -18.6135330, 35.2921753, -19.0125427, 35.1322556, -53.7457848, 54.3047180
1: -20.9900990, 32.1215439, -21.4134941, 32.7879028, -53.7779999, 53.5350304
2: -21.5068703, 31.5812416, -21.9236660, 32.1679916, -53.6748619, 53.5049019
3: -25.8637333, 37.0863228, -26.3699760, 37.8068047, -63.6705399, 63.4562988
4: -24.3186264, 35.1811447, -24.7712917, 35.7631111, -60.0817375, 59.9524345

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8741262, upper bound: 53.7122626
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8741262, upper bound: 53.7122626
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -18.6135330, 35.2921753, -23.5083370, 43.5609932, -62.1745186, 58.8005142
1: -20.9900990, 32.1215439, -26.5329418, 40.2712250, -61.2613220, 58.6544876
2: -21.5068703, 31.5812416, -27.0840607, 39.4476700, -60.9545250, 58.6653023
3: -25.8637333, 37.0863228, -32.7493973, 46.5774879, -72.4412155, 69.8357239
4: -24.3186264, 35.1811447, -30.5417747, 44.2039719, -68.5225983, 65.7229156

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8741262, upper bound: 54.2409729
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8741262, upper bound: 53.7122626
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -25.4063072, 46.8272247, -19.0125427, 35.1322556, -60.5385475, 65.8397675
1: -28.6462059, 43.1931686, -21.4134941, 32.7879028, -61.4341087, 64.6066437
2: -29.2610550, 42.2853508, -21.9236660, 32.1679916, -61.4290428, 64.2089996
3: -35.3114586, 50.0333862, -26.3699760, 37.8068047, -73.1182632, 76.4033661
4: -32.9586563, 47.5135345, -24.7712917, 35.7631111, -68.7217484, 72.2848129

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -25.4063072, 46.8272247, -23.5083370, 43.5609932, -68.9673004, 70.3355637
1: -28.6462059, 43.1931686, -26.5329418, 40.2712250, -68.9174271, 69.7260971
2: -29.2610550, 42.2853508, -27.0840607, 39.4476700, -68.7087250, 69.3694077
3: -35.3114586, 50.0333862, -32.7493973, 46.5774879, -81.8889465, 82.7827835
4: -32.9586563, 47.5135345, -30.5417747, 44.2039719, -77.1626205, 78.0553131

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2404651
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2404651
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17.9468613, 34.1551132, -21.6481285, 39.5172691, -57.4641266, 55.8032341
1: -20.2491722, 31.4106674, -24.3567982, 36.7261505, -56.9753227, 55.7674561
2: -20.7486954, 30.8505535, -24.9315796, 35.9881516, -56.7368469, 55.7821236
3: -24.9880733, 36.2773743, -29.9461422, 42.4385223, -67.4265976, 66.2235184
4: -23.5056725, 34.2449112, -28.0547447, 40.2319031, -63.7375641, 62.2996559

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9783601, upper bound: 53.5718064
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9783601, upper bound: 53.5718139
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17.9468613, 34.1551132, -25.8506546, 47.4542923, -65.4011536, 60.0057602
1: -20.2491722, 31.4106674, -29.1244144, 43.7585106, -64.0076828, 60.5350723
2: -20.7486954, 30.8505535, -29.7637978, 42.8362465, -63.5849419, 60.6143417
3: -24.9880733, 36.2773743, -35.8765335, 50.7101250, -75.6981964, 72.1539078
4: -23.5056725, 34.2449112, -33.4723434, 48.1443787, -71.6500473, 67.7172470

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9783602, upper bound: 54.1659864
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9783602, upper bound: 54.1660698
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -20.4113770, 38.2129784, -21.6481285, 39.5172691, -59.9286461, 59.8611031
1: -22.9685669, 35.0723991, -24.3567982, 36.7261505, -59.6947174, 59.4291801
2: -23.5585613, 34.4084091, -24.9315796, 35.9881516, -59.5466957, 59.3399734
3: -28.2634182, 40.5897751, -29.9461422, 42.4385223, -70.7019424, 70.5359192
4: -26.5753155, 38.3587837, -28.0547447, 40.2319031, -66.8071976, 66.4135284

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -20.4113770, 38.2129784, -25.8506546, 47.4542923, -67.8656693, 64.0636215
1: -22.9685669, 35.0723991, -29.1244144, 43.7585106, -66.7270813, 64.1968002
2: -23.5585613, 34.4084091, -29.7637978, 42.8362465, -66.3948059, 64.1722107
3: -28.2634182, 40.5897751, -35.8765335, 50.7101250, -78.9735336, 76.4663086
4: -26.5753155, 38.3587837, -33.4723434, 48.1443787, -74.7196808, 71.8311157

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067
time: 0.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.37 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2238161, upper bound: 54.1533464
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2368939, upper bound: 54.2034040
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2238161, upper bound: 54.1533464
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2368939, upper bound: 54.2034040
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7137535, upper bound: 53.5314479
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5732737, upper bound: 53.3574252
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7137535, upper bound: 53.5314479
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5732737, upper bound: 53.3574252
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2186915, upper bound: 54.1707650
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2265096, upper bound: 54.1726818
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2186915, upper bound: 54.1707650
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2265096, upper bound: 54.1726818
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5259819
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7508023, upper bound: 53.8933997
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5259819
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7508023, upper bound: 53.8933997
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.9204798, upper bound: 53.0603758
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5634923, upper bound: 53.8864395
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.9204798, upper bound: 53.0603758
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5634923, upper bound: 53.8864395
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1110167, upper bound: 54.1683255
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1902880, upper bound: 54.1716890
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1110167, upper bound: 54.1683255
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1902880, upper bound: 54.1716890
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.6073332, upper bound: 53.9251114
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5432133, upper bound: 53.9087205
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.6073332, upper bound: 53.9251113
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5432133, upper bound: 53.9087205
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.7583243, upper bound: 52.7925034
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.0515877, upper bound: 52.9141563
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7650425, upper bound: 53.3960800
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9750785, upper bound: 53.5673683
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.4628825, upper bound: 52.6763389
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.3065274, upper bound: 52.6195791
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.8539414, upper bound: 52.8583024
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.6903222, upper bound: 52.8214052
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8253629, upper bound: 53.6542159
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2179532, upper bound: 54.1085246
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8253629, upper bound: 53.6542159
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2179532, upper bound: 54.1085246
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5168676
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7490094, upper bound: 53.8883722
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5717265, upper bound: 53.5168676
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.7490094, upper bound: 53.8883724
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.9409143, upper bound: 53.0935559
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8759553, upper bound: 53.5634925
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.9409143, upper bound: 53.0935559
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8759553, upper bound: 53.5634925
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.9153733, upper bound: 53.0435062
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5258719, upper bound: 53.5258721
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -52.9153733, upper bound: 53.0435062
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5258719, upper bound: 53.5258721
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9698747, upper bound: 54.1400144
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8979122, upper bound: 53.9470216
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9698747, upper bound: 54.1400144
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8979122, upper bound: 53.9470218
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5544349, upper bound: 53.6848892
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5139289, upper bound: 53.6943285
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5544349, upper bound: 53.6848892
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5139290, upper bound: 53.6943283
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2104433, upper bound: 53.9549150
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.5837629, upper bound: 53.4268794
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.0974909, upper bound: 53.8838981
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.0974909, upper bound: 53.8838981
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1054048, upper bound: 53.9621226
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2183512, upper bound: 54.1745227
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2183512, upper bound: 54.1745227
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2183512, upper bound: 54.1745227
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.2412284, upper bound: 54.1745227
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -54.1642077, upper bound: 54.1717398
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8741262, upper bound: 53.7122626
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8741262, upper bound: 53.7122626
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8741262, upper bound: 54.2409729
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8741262, upper bound: 53.7122626
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673046, upper bound: 53.7102884
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2404651
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673048, upper bound: 54.2404651
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9783601, upper bound: 53.5718064
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9783601, upper bound: 53.5718139
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9783602, upper bound: 54.1659864
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.9783602, upper bound: 54.1660698
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673046, upper bound: 53.5648394
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.37
Output dim: 0, lower bound: -53.8673048, upper bound: 54.1633067

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.8360004, 21.5876408, -11.2628803, 22.3128376, -33.1488342, 32.8505211
1: -12.2428493, 19.7818031, -12.7340174, 20.4601612, -32.7030106, 32.5158195
2: -12.6421089, 19.5397930, -13.1224794, 20.2113457, -32.8534546, 32.6622696
3: -15.0583124, 22.5945473, -15.6772995, 23.3713398, -38.4296532, 38.2718430
4: -14.4506550, 21.2589779, -14.9822273, 22.0476837, -36.4983368, 36.2412033

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2415351, upper bound: 54.2415351
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2415351, upper bound: 54.2488080
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.1134596, 22.0230827, -11.4045887, 22.5669975, -33.6804504, 33.4276733
1: -12.5614967, 20.1963253, -12.8947296, 20.7004795, -33.2619781, 33.0910568
2: -12.9547071, 19.9538803, -13.2829790, 20.4475098, -33.4022141, 33.2368584
3: -15.4615765, 23.0673733, -15.8785458, 23.6497173, -39.1112938, 38.9459114
4: -14.7867708, 21.7569885, -15.1618385, 22.3195095, -37.1062813, 36.9188271

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2488080, upper bound: 54.2415930
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2488080, upper bound: 54.2553633
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.8360004, 21.5876408, -13.8961887, 26.8542957, -37.6902962, 35.4838257
1: -12.2428493, 19.7818031, -15.6723557, 24.4922180, -36.7350655, 35.4541512
2: -12.6421089, 19.5397930, -16.1136703, 24.1579323, -36.8000374, 35.6534615
3: -15.0583124, 22.5945473, -19.2553501, 28.0586624, -43.1169739, 41.8498917
4: -14.4506550, 21.2589779, -18.2475376, 26.6046658, -41.0553169, 39.5065155

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2208185, upper bound: 54.1479983
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2208185, upper bound: 54.1533464
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.1134596, 22.0230827, -14.0177517, 27.0706272, -38.1840858, 36.0408325
1: -12.5614967, 20.1963253, -15.8099871, 24.6929855, -37.2544785, 36.0063095
2: -12.9547071, 19.9538803, -16.2508926, 24.3535080, -37.3082085, 36.2047653
3: -15.4615765, 23.0673733, -19.4277878, 28.2923222, -43.7538948, 42.4951591
4: -14.7867708, 21.7569885, -18.4020214, 26.8302002, -41.6169701, 40.1590004

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2368939, upper bound: 54.1977671
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2354563, upper bound: 54.2034040
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16.3784657, 30.4600277, -10.9949780, 21.8330498, -38.2115173, 41.4549980
1: -18.4226360, 28.3888454, -12.4343500, 20.0266819, -38.4493179, 40.8231964
2: -18.9195595, 27.9135056, -12.8220434, 19.7868080, -38.7063675, 40.7355461
3: -22.6590843, 32.6828880, -15.3129072, 22.8656693, -45.5247536, 47.9957886
4: -21.3762970, 30.8894005, -14.6527243, 21.5710831, -42.9473724, 45.5421257

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7149317, upper bound: 54.0276529
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7641487, upper bound: 54.1819225
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20.0840397, 36.4824753, -10.7595930, 21.3877697, -41.4718094, 47.2420692
1: -22.5328808, 33.5295410, -12.1630411, 19.5631580, -42.0960388, 45.6925812
2: -23.1216640, 32.9704628, -12.5547485, 19.3355541, -42.4572182, 45.5252113
3: -27.6449432, 38.7795372, -14.9651413, 22.3377819, -49.9827271, 53.7446747
4: -25.7313786, 36.9697075, -14.3371964, 21.0632057, -46.7945862, 51.3069038

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6553325, upper bound: 54.0001577
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6553325, upper bound: 54.0123890
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.3784657, 30.4600277, -13.5955238, 26.3417683, -42.7202339, 44.0555496
1: -18.4226360, 28.3888454, -15.3357601, 24.0065365, -42.4291725, 43.7245903
2: -18.9195595, 27.9135056, -15.7737818, 23.6885147, -42.6080742, 43.6872826
3: -22.6590843, 32.6828880, -18.8418388, 27.4867001, -50.1457825, 51.5247192
4: -21.3762970, 30.8894005, -17.8674202, 26.0738373, -47.4501343, 48.7568207

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6573510, upper bound: 53.4370632
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7061597, upper bound: 53.5241806
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.0840397, 36.4824753, -13.4083300, 25.9927711, -46.0768089, 49.8908043
1: -22.5328808, 33.5295410, -15.1148958, 23.6652126, -46.1980934, 48.6444359
2: -23.1216640, 32.9704628, -15.5644808, 23.3562012, -46.4778671, 48.5349350
3: -27.6449432, 38.7795372, -18.5560970, 27.1043930, -54.7493362, 57.3356323
4: -25.7313786, 36.9697075, -17.6235619, 25.6863365, -51.4177094, 54.5932693

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5688997, upper bound: 53.3455006
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5688997, upper bound: 53.3574252
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.0300674, 20.1934052, -16.5780373, 31.9165230, -41.9465790, 36.7714424
1: -11.3704586, 18.4050407, -18.7406483, 29.0654297, -40.4358902, 37.1456833
2: -11.7297525, 18.1919785, -19.1804962, 28.6155891, -40.3453407, 37.3724670
3: -13.9982967, 20.9615364, -23.1438141, 33.4699211, -47.4682159, 44.1053505
4: -13.4208279, 19.7866077, -21.7336960, 31.7478294, -45.1686554, 41.5202942

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2348575, upper bound: 54.2406217
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2348575, upper bound: 54.2406217
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.9988451, 21.9234638, -15.9009981, 30.7976341, -41.7964783, 37.8244629
1: -12.4454994, 20.0434628, -17.9824200, 27.9855003, -40.4309998, 38.0258827
2: -12.8339987, 19.7845306, -18.4096928, 27.5655785, -40.3995781, 38.1942101
3: -15.2732391, 22.8486443, -22.2011490, 32.1959267, -47.4691620, 45.0497856
4: -14.6963940, 21.5434532, -20.8762360, 30.5448380, -45.2412300, 42.4196892

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2348575, upper bound: 54.2406217
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2348575, upper bound: 54.2406217
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.0300674, 20.1934052, -18.8830833, 35.7308578, -45.7609253, 39.0764885
1: -11.3704586, 18.4050407, -21.2896595, 32.5061417, -43.8766022, 39.6946983
2: -11.7297525, 18.1919785, -21.8136520, 31.9565125, -43.6862564, 40.0056267
3: -13.9982967, 20.9615364, -26.2257919, 37.5368881, -51.5351868, 47.1873207
4: -13.4208279, 19.7866077, -24.6326466, 35.6247749, -49.0456009, 44.4192505

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2186915, upper bound: 54.0785502
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2186915, upper bound: 54.1707650
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.9988451, 21.9234638, -18.1267738, 34.4667511, -45.4655952, 40.0502396
1: -12.4454994, 20.0434628, -20.4439983, 31.2905483, -43.7360458, 40.4874611
2: -12.8339987, 19.7845306, -20.9495163, 30.7760448, -43.6100426, 40.7340393
3: -15.2732391, 22.8486443, -25.1775074, 36.0936394, -51.3668709, 48.0261459
4: -14.6963940, 21.5434532, -23.6641579, 34.2664719, -48.9628601, 45.2076111

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2265096, upper bound: 54.1726818
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2265096, upper bound: 54.0791583
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2265096, upper bound: 54.1726818
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17.7713623, 32.9331398, -16.3490982, 31.5172882, -49.2886505, 49.2822380
1: -20.0100880, 30.6770668, -18.4804630, 28.6747055, -48.6847878, 49.1575317
2: -20.5026169, 30.1220856, -18.9153862, 28.2372074, -48.7398224, 49.0374680
3: -24.6291695, 35.3657951, -22.8199196, 33.0142136, -57.6433830, 58.1856995
4: -23.1756878, 33.4063530, -21.4326401, 31.3165340, -54.4922218, 54.8389816

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6915111, upper bound: 53.9509787
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6915111, upper bound: 54.0540768
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.9984283, 33.2768784, -16.4982262, 31.7712708, -49.7696991, 49.7750931
1: -20.2728310, 31.0072956, -18.6489811, 28.9342461, -49.2070770, 49.6562767
2: -20.7596989, 30.4490814, -19.0906982, 28.4886093, -49.2483063, 49.5397797
3: -24.9616203, 35.7401962, -23.0291786, 33.3176575, -58.2792778, 58.7693748
4: -23.4484921, 33.8082314, -21.6317406, 31.5982342, -55.0467148, 55.4399719

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7687160, upper bound: 54.1843421
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7687836, upper bound: 54.1832850
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17.7713623, 32.9331398, -18.6499748, 35.3182220, -53.0895805, 51.5831146
1: -20.0100880, 30.6770668, -21.0242252, 32.1035309, -52.1136017, 51.7012863
2: -20.5026169, 30.1220856, -21.5425854, 31.5692635, -52.0718765, 51.6646652
3: -24.6291695, 35.3657951, -25.8963699, 37.0655212, -61.6946907, 61.2621651
4: -23.1756878, 33.4063530, -24.3245735, 35.1811371, -58.3568192, 57.7309227

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.9984283, 33.2768784, -18.7998066, 35.5804787, -53.5789032, 52.0766830
1: -20.2728310, 31.0072956, -21.1935463, 32.3684998, -52.6413307, 52.2008438
2: -20.7596989, 30.4490814, -21.7197781, 31.8235092, -52.5832062, 52.1688576
3: -24.9616203, 35.7401962, -26.1050587, 37.3762932, -62.3379097, 61.8452530
4: -23.4484921, 33.8082314, -24.5261517, 35.4664917, -58.9149857, 58.3343811

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7421201, upper bound: 53.8825636
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7427279, upper bound: 53.8859068
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.6254129, 24.7713547, -11.4906597, 22.7267227, -35.3521347, 36.2620010
1: -14.2655630, 22.4507809, -12.9931536, 20.8490124, -35.1145630, 35.4439354
2: -14.6793137, 22.1696510, -13.3803749, 20.5927849, -35.2720985, 35.5500221
3: -17.5229568, 25.6474609, -16.0014496, 23.8214302, -41.3443871, 41.6489067
4: -16.6304893, 24.3550930, -15.2732220, 22.4852295, -39.1157188, 39.6283112

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1189438, upper bound: 54.2227536
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1189438, upper bound: 54.2284878
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.6885767, 26.5523262, -10.7964926, 21.5563736, -35.2449493, 37.3488159
1: -15.4345446, 24.2497425, -12.2152195, 19.7483559, -35.1828918, 36.4649620
2: -15.8949060, 23.9035759, -12.5987215, 19.5073872, -35.4022942, 36.5022964
3: -18.8986511, 27.7512016, -15.0351658, 22.5334167, -41.4320602, 42.7863617
4: -18.0523510, 26.2602692, -14.4176197, 21.2454662, -39.2978172, 40.6778870

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1994394, upper bound: 54.2227536
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1994394, upper bound: 54.2284878
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.6254129, 24.7713547, -14.1123228, 27.2404823, -39.8658943, 38.8836784
1: -14.2655630, 22.4507809, -15.9184780, 24.8546124, -39.1201591, 38.3692589
2: -14.6793137, 22.1696510, -16.3577156, 24.5101395, -39.1894531, 38.5273628
3: -17.5229568, 25.6474609, -19.5635452, 28.4796600, -46.0026169, 45.2110062
4: -16.6304893, 24.3550930, -18.5248909, 27.0098228, -43.6403122, 42.8799820

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1149433, upper bound: 54.1149433
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1149433, upper bound: 54.1971245
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.6885767, 26.5523262, -13.3975296, 26.0639248, -39.7525024, 39.9498444
1: -15.4345446, 24.2497425, -15.1157265, 23.7213173, -39.1558609, 39.3654709
2: -15.8949060, 23.9035759, -15.5496693, 23.4054508, -39.3003578, 39.4532433
3: -18.8986511, 27.7512016, -18.5648499, 27.1473198, -46.0459671, 46.3160362
4: -18.0523510, 26.2602692, -17.6258774, 25.7443542, -43.7967072, 43.8861465

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1971245, upper bound: 54.1182978
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1971245, upper bound: 54.2004791
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.3689442, 28.6628838, -10.2119761, 20.3488426, -35.7177811, 38.8748589
1: -17.2642746, 25.5136528, -11.5452738, 18.5013504, -35.7656250, 37.0589256
2: -17.7097759, 25.1806984, -11.9215336, 18.3030338, -36.0128098, 37.1022339
3: -21.0907421, 29.2756767, -14.1903076, 21.0924492, -42.1831894, 43.4659843
4: -19.5813313, 28.0416107, -13.5824699, 19.9194107, -39.5007362, 41.6240654

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7925034, upper bound: 52.7583243
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9141563, upper bound: 53.0515877
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.5253487, 35.8569069, -11.4906597, 22.7267227, -42.2520714, 47.3475609
1: -21.9945335, 33.1004257, -12.9931536, 20.8490124, -42.8435402, 46.0935783
2: -22.4913483, 32.4782181, -13.3803749, 20.5927849, -43.0841331, 45.8585892
3: -27.0583725, 38.1645317, -16.0014496, 23.8214302, -50.8798027, 54.1659775
4: -25.2483902, 36.2808876, -15.2732220, 22.4852295, -47.7336159, 51.5541039

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3960800, upper bound: 53.7650425
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5673683, upper bound: 53.9750785
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.3689442, 28.6628838, -12.8584404, 25.0414543, -40.4104004, 41.5213242
1: -17.2642746, 25.5136528, -14.5007305, 22.6506462, -39.9149208, 40.0143814
2: -17.7097759, 25.1806984, -14.9319248, 22.3777809, -40.0875473, 40.1126137
3: -21.0907421, 29.2756767, -17.7893562, 25.9027176, -46.9934578, 47.0650291
4: -19.5813313, 28.0416107, -16.8650284, 24.6002502, -44.1815681, 44.9066277

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5405901, upper bound: 52.5169657
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5405906, upper bound: 53.0603758
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.5253487, 35.8569069, -14.1123228, 27.2404823, -46.7658310, 49.9692307
1: -21.9945335, 33.1004257, -15.9184780, 24.8546124, -46.8491402, 49.0189018
2: -22.4913483, 32.4782181, -16.3577156, 24.5101395, -47.0014877, 48.8359299
3: -27.0583725, 38.1645317, -19.5635452, 28.4796600, -55.5380325, 57.7280769
4: -25.2483902, 36.2808876, -18.5248909, 27.0098228, -52.2582054, 54.8057747

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8745826, upper bound: 52.8917663
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8745826, upper bound: 53.8864396
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.6254129, 24.7713547, -16.5780373, 31.9165230, -44.5419312, 41.3493843
1: -14.2655630, 22.4507809, -18.7406483, 29.0654297, -43.3309860, 41.1914291
2: -14.6793137, 22.1696510, -19.1804962, 28.6155891, -43.2949028, 41.3501396
3: -17.5229568, 25.6474609, -23.1438141, 33.4699211, -50.9928780, 48.7912750
4: -16.6304893, 24.3550930, -21.7336960, 31.7478294, -48.3783188, 46.0887833

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0955347, upper bound: 54.2089549
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0955347, upper bound: 54.2294621
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.6885767, 26.5523262, -15.9009981, 30.7976341, -44.4862099, 42.4533234
1: -15.4345446, 24.2497425, -17.9824200, 27.9855003, -43.4200439, 42.2321625
2: -15.8949060, 23.9035759, -18.4096928, 27.5655785, -43.4604836, 42.3132668
3: -18.8986511, 27.7512016, -22.2011490, 32.1959267, -51.0945740, 49.9523468
4: -18.0523510, 26.2602692, -20.8762360, 30.5448380, -48.5971909, 47.1365051

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1759277, upper bound: 54.2089549
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1759277, upper bound: 54.2294621
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.6254129, 24.7713547, -18.8832836, 35.7310181, -48.3564301, 43.6546364
1: -14.2655630, 22.4507809, -21.2898827, 32.5063171, -46.7718811, 43.7406616
2: -14.6793137, 22.1696510, -21.8138657, 31.9566841, -46.6359978, 43.9835052
3: -17.5229568, 25.6474609, -26.2260628, 37.5371437, -55.0601006, 51.8735237
4: -16.6304893, 24.3550930, -24.6329060, 35.6249886, -52.2554779, 48.9879990

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0955347, upper bound: 54.0737587
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0955347, upper bound: 54.1683255
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.6885767, 26.5523262, -18.1268578, 34.4668159, -48.1553879, 44.6791840
1: -15.4345446, 24.2497425, -20.4440880, 31.2906189, -46.7251587, 44.6938324
2: -15.8949060, 23.9035759, -20.9496002, 30.7761173, -46.6710243, 44.8531761
3: -18.8986511, 27.7512016, -25.1776142, 36.0937386, -54.9923897, 52.9288101
4: -18.0523510, 26.2602692, -23.6642647, 34.2665520, -52.3189011, 49.9245338

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1710252, upper bound: 54.1689451
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1759170, upper bound: 54.0770622
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1759170, upper bound: 54.1716890
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18.2548027, 33.8085480, -15.6957397, 30.3983841, -48.6531830, 49.5042877
1: -20.5351543, 31.0225201, -17.7389965, 27.6324158, -48.1675720, 48.7615089
2: -21.0312881, 30.4730453, -18.1735077, 27.2275696, -48.2588577, 48.6465492
3: -25.2424526, 35.7639313, -21.8974361, 31.7953587, -57.0378113, 57.6613693
4: -23.5571022, 33.9298935, -20.5930538, 30.1250286, -53.6821289, 54.5229378

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5469935, upper bound: 54.0001027
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5469935, upper bound: 54.0001027
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.5364857, 35.8864326, -16.4962025, 31.7572212, -51.2937088, 52.3826294
1: -22.0320759, 33.2459717, -18.6485863, 28.9061909, -50.9382668, 51.8945541
2: -22.5131683, 32.6049652, -19.0828991, 28.4636021, -50.9767685, 51.6878662
3: -27.1324921, 38.3395615, -23.0303555, 33.2877350, -60.4202271, 61.3699150
4: -25.3367653, 36.4245148, -21.6195145, 31.5818253, -56.9185867, 58.0440292

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3761299, upper bound: 53.6679323
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5412882, upper bound: 53.9951455
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18.2548027, 33.8085480, -17.9812584, 34.1813431, -52.4361420, 51.7898064
1: -20.5351543, 31.0225201, -20.2611008, 31.0387058, -51.5738602, 51.2836189
2: -21.0312881, 30.4730453, -20.7801304, 30.5355873, -51.5668755, 51.2531738
3: -25.2424526, 35.7639313, -24.9430485, 35.8084412, -61.0508957, 60.7069778
4: -23.5571022, 33.9298935, -23.4581680, 33.9454727, -57.5025673, 57.3880539

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5423200, upper bound: 53.9077946
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5423200, upper bound: 53.9077946
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.5364857, 35.8864326, -18.7972221, 35.5640411, -55.1005249, 54.6836548
1: -22.0320759, 33.2459717, -21.1929245, 32.3397598, -54.3718338, 54.4388885
2: -22.5131683, 32.6049652, -21.7110214, 31.7997379, -54.3129044, 54.3159866
3: -27.1324921, 38.3395615, -26.1067562, 37.3454742, -64.4779663, 64.4463043
4: -25.3367653, 36.4245148, -24.5129776, 35.4505463, -60.7873116, 60.9374924

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5423200, upper bound: 53.9077946
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5423200, upper bound: 53.9087205
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.5774565, 19.2125511, -16.9829941, 31.3730316, -40.9504890, 36.1955452
1: -10.8156414, 17.4382668, -19.0203743, 28.2632179, -39.0788574, 36.4586411
2: -11.2022562, 17.2519112, -19.5612450, 27.8576374, -39.0598869, 36.8131523
3: -13.2681999, 19.8705559, -23.1827526, 32.5487022, -45.8169022, 43.0533066
4: -12.7744589, 18.7039108, -21.7704582, 30.9391937, -43.7136497, 40.4743652

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7425129, upper bound: 52.7894365
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5715199, upper bound: 52.7169359
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7366719, upper bound: 52.7851755
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.8399153, 19.6342201, -17.1027946, 31.5790787, -41.4189949, 36.7370110
1: -11.1185350, 17.8389111, -19.1552391, 28.4574852, -39.5760193, 36.9941444
2: -11.4994097, 17.6539230, -19.6969662, 28.0452194, -39.5446281, 37.3508873
3: -13.6524363, 20.3254280, -23.3504143, 32.7761650, -46.4286003, 43.6758308
4: -13.0950832, 19.1838531, -21.9227428, 31.1540012, -44.2490768, 41.1065903

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0099141, upper bound: 52.8791134
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8215775, upper bound: 52.7956707
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8215775, upper bound: 52.9141563
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.8360004, 21.5876408, -21.2726173, 38.9270477, -49.7630463, 42.8602562
1: -12.2428493, 19.7818031, -23.9310455, 36.1842804, -48.4271317, 43.7128410
2: -12.6421089, 19.5397930, -24.5047970, 35.4641380, -48.1062469, 44.0445900
3: -15.0583124, 22.5945473, -29.4102497, 41.8046417, -56.8629532, 52.0047951
4: -14.4506550, 21.2589779, -27.6048164, 39.5976868, -54.0483398, 48.8637924

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7650425, upper bound: 53.3960800
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7650425, upper bound: 53.3960800
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.1134596, 22.0230827, -21.3859577, 39.1226997, -50.2361526, 43.4090347
1: -12.5614967, 20.1963253, -24.0589752, 36.3679619, -48.9294586, 44.2552948
2: -12.9547071, 19.9538803, -24.6338005, 35.6415291, -48.5962296, 44.5876770
3: -15.4615765, 23.0673733, -29.5687580, 42.0203819, -57.4819527, 52.6361198
4: -14.7867708, 21.7569885, -27.7481689, 39.8049126, -54.5916824, 49.5051537

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9750785, upper bound: 53.5673683
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9750785, upper bound: 53.5673683
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.1389294, 28.2651367, -16.8012352, 31.0372276, -46.1761551, 45.0663719
1: -17.0202332, 26.1739502, -18.8110619, 27.9617958, -44.9820290, 44.9850121
2: -17.5021820, 25.7872162, -19.3534012, 27.5660763, -45.0682602, 45.1406174
3: -20.8948612, 30.0859852, -22.9215908, 32.1980400, -53.0928917, 53.0075760
4: -19.7141342, 28.4867668, -21.5359077, 30.6017914, -50.3159103, 50.0226669

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3065274, upper bound: 52.6195791
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3065274, upper bound: 52.6195791
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.0064125, 34.6231956, -16.6659069, 30.7871857, -49.7935982, 51.2891006
1: -21.2955933, 31.6514530, -18.6627922, 27.7282543, -49.0238495, 50.3142395
2: -21.8900127, 31.1610756, -19.1977539, 27.3406639, -49.2306747, 50.3588295
3: -26.0911713, 36.5952568, -22.7420864, 31.9274864, -58.0186577, 59.3373413
4: -24.3030682, 34.9013863, -21.3561211, 30.3509293, -54.6539993, 56.2575035

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3065274, upper bound: 52.6195791
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3065274, upper bound: 52.6195791
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.3784657, 30.4600277, -20.9819489, 38.3859253, -54.7643890, 51.4419785
1: -18.4226360, 28.3888454, -23.5970230, 35.6882935, -54.1109314, 51.9858589
2: -18.9195595, 27.9135056, -24.1718922, 34.9841347, -53.9036942, 52.0853882
3: -22.6590843, 32.6828880, -28.9945202, 41.2271538, -63.8862381, 61.6774063
4: -21.3762970, 30.8894005, -27.2293587, 39.0446281, -60.4209251, 58.1187553

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8539414, upper bound: 52.8583024
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8539414, upper bound: 52.8583024
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.0840397, 36.4824753, -20.4795246, 37.4699936, -57.5540314, 56.9619980
1: -22.5328808, 33.5295410, -23.0349236, 34.7603645, -57.2932434, 56.5644646
2: -23.1216640, 32.9704628, -23.5967007, 34.1011887, -57.2228546, 56.5671616
3: -27.6449432, 38.7795372, -28.2971268, 40.1282921, -67.7732391, 67.0766449
4: -25.7313786, 36.9697075, -26.5550137, 38.0489464, -63.7803192, 63.5247192

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6903222, upper bound: 52.8214052
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6903222, upper bound: 52.8214052
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.8360004, 21.5876408, -23.2890110, 43.1814270, -54.0174255, 44.8766518
1: -12.2428493, 19.7818031, -26.2827377, 39.9056931, -52.1485443, 46.0645332
2: -12.6421089, 19.5397930, -26.8325958, 39.0962715, -51.7383804, 46.3723869
3: -15.0583124, 22.5945473, -32.4394531, 46.1383171, -61.1966286, 55.0339890
4: -14.4506550, 21.2589779, -30.2505875, 43.7934189, -58.2440720, 51.5095634

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0553361, upper bound: 54.0948886
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0561912, upper bound: 54.0845178
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0533360, upper bound: 54.0123169
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.1134596, 22.0230827, -23.4260082, 43.4121666, -54.5256195, 45.4490891
1: -12.5614967, 20.1963253, -26.4383926, 40.1334152, -52.6949120, 46.6347122
2: -12.9547071, 19.9538803, -26.9904499, 39.3144760, -52.2691841, 46.9443207
3: -15.4615765, 23.0673733, -32.6322899, 46.4192848, -61.8808594, 55.6996536
4: -14.7867708, 21.7569885, -30.4366264, 44.0493088, -58.8360748, 52.1936073

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2373802, upper bound: 54.1964032
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2241478, upper bound: 54.1746006
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0740152, upper bound: 54.0163067
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.8360004, 21.5876408, -25.6314487, 47.0678558, -57.9038506, 47.2190895
1: -12.2428493, 19.7818031, -28.8744240, 43.3900223, -55.6328735, 48.6562271
2: -12.6421089, 19.5397930, -29.5125008, 42.4824219, -55.1245308, 49.0522881
3: -15.0583124, 22.5945473, -35.5681419, 50.2687759, -65.3270874, 58.1626854
4: -14.4506550, 21.2589779, -33.1815109, 47.7324409, -62.1830978, 54.4404907

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8231313, upper bound: 53.5662774
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6344056, upper bound: 53.6202508
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8197389, upper bound: 53.6522343
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.1134596, 22.0230827, -25.7667961, 47.3039246, -58.4173813, 47.7898712
1: -12.5614967, 20.1963253, -29.0279064, 43.6188202, -56.1803093, 49.2242317
2: -12.9547071, 19.9538803, -29.6685352, 42.7012863, -55.6559944, 49.6224098
3: -15.4615765, 23.0673733, -35.7568512, 50.5498276, -66.0113907, 58.8242226
4: -14.7867708, 21.7569885, -33.3656311, 47.9869385, -62.7737083, 55.1226196

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0755616, upper bound: 53.6463569
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2137533, upper bound: 54.1032197
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1391587, upper bound: 53.8385680
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17.7713623, 32.9331398, -23.2890110, 43.1814270, -60.9527702, 56.2221527
1: -20.0100880, 30.6770668, -26.2827377, 39.9056931, -59.9157639, 56.9598045
2: -20.5026169, 30.1220856, -26.8325958, 39.0962715, -59.5988808, 56.9546661
3: -24.6291695, 35.3657951, -32.4394531, 46.1383171, -70.7674866, 67.8052444
4: -23.1756878, 33.4063530, -30.2505875, 43.7934189, -66.9691086, 63.6569405

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4433255, upper bound: 53.9368903
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6846680, upper bound: 54.0481673
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.9984283, 33.2768784, -23.4260082, 43.4121666, -61.4105873, 56.7028847
1: -20.2728310, 31.0072956, -26.4383926, 40.1334152, -60.4062462, 57.4456863
2: -20.7596989, 30.4490814, -26.9904499, 39.3144760, -60.0741730, 57.4395218
3: -24.9616203, 35.7401962, -32.6322899, 46.4192848, -71.3809052, 68.3724823
4: -23.4484921, 33.8082314, -30.4366264, 44.0493088, -67.4977875, 64.2448502

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7687044, upper bound: 54.1843421
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7671355, upper bound: 54.1811915
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17.7713623, 32.9331398, -25.6314487, 47.0678558, -64.8392029, 58.5645905
1: -20.0100880, 30.6770668, -28.8744240, 43.3900223, -63.4000854, 59.5514908
2: -20.5026169, 30.1220856, -29.5125008, 42.4824219, -62.9850388, 59.6345634
3: -24.6291695, 35.3657951, -35.5681419, 50.2687759, -74.8979492, 70.9339142
4: -23.1756878, 33.4063530, -33.1815109, 47.7324409, -70.9081268, 66.5878601

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3255226, upper bound: 53.3925039
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5531650, upper bound: 53.5087167
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17.9984283, 33.2768784, -25.7667961, 47.3039246, -65.3023376, 59.0436745
1: -20.2728310, 31.0072956, -29.0279064, 43.6188202, -63.8916512, 60.0351906
2: -20.7596989, 30.4490814, -29.6685352, 42.7012863, -63.4609833, 60.1176147
3: -24.9616203, 35.7401962, -35.7568512, 50.5498276, -75.5114441, 71.4970474
4: -23.4484921, 33.8082314, -33.3656311, 47.9869385, -71.4354324, 67.1738586

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7406219, upper bound: 53.8807577
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7104777, upper bound: 53.6739574
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.4541225, 18.7644768, -17.6597652, 32.7593842, -42.2135048, 36.4242325
1: -10.5632315, 16.0994434, -19.8761311, 30.3878574, -40.9510880, 35.9755707
2: -10.9691639, 15.9714804, -20.3709641, 29.8667965, -40.8359604, 36.3424454
3: -12.7866840, 18.3013859, -24.4481449, 34.9785995, -47.7652817, 42.7495232
4: -12.1467257, 17.4748402, -22.9578114, 33.1388779, -45.2855988, 40.4326477

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9376616, upper bound: 53.0708654
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7020815, upper bound: 52.5302993
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.2677364, 25.7574368, -19.0125427, 35.1322556, -48.3999825, 44.7699776
1: -14.9546404, 23.3805790, -21.4134941, 32.7879028, -47.7425423, 44.7940636
2: -15.3958588, 23.0812187, -21.9236660, 32.1679916, -47.5638504, 45.0048752
3: -18.3520355, 26.7642899, -26.3699760, 37.8068047, -56.1588402, 53.1342583
4: -17.4056129, 25.3842506, -24.7712917, 35.7631111, -53.1687241, 50.1555405

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4705896, upper bound: 53.6672142
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3075328, upper bound: 53.5297176
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.4541225, 18.7644768, -20.3451557, 37.2922020, -46.7463226, 39.1096306
1: -10.5632315, 16.0994434, -22.8790359, 34.4719276, -45.0351601, 38.9784775
2: -10.9691639, 15.9714804, -23.4392948, 33.8204613, -44.7896271, 39.4107742
3: -12.7866840, 18.3013859, -28.1053257, 39.7860641, -52.5727463, 46.4067116
4: -12.1467257, 17.4748402, -26.3313980, 37.7500992, -49.8968201, 43.8062363

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5298442, upper bound: 52.5494686
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5298442, upper bound: 53.0935559
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.2677364, 25.7574368, -21.6481285, 39.5172691, -52.7850037, 47.4055634
1: -14.9546404, 23.3805790, -24.3567982, 36.7261505, -51.6807899, 47.7373772
2: -15.3958588, 23.0812187, -24.9315796, 35.9881516, -51.3840103, 48.0127907
3: -18.3520355, 26.7642899, -29.9461422, 42.4385223, -60.7905502, 56.7104263
4: -17.4056129, 25.3842506, -28.0547447, 40.2319031, -57.6375160, 53.4389954

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8929352, upper bound: 52.8873252
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8929352, upper bound: 53.5634925
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.3688536, 28.6628151, -17.6597652, 32.7593842, -48.1282387, 46.3225784
1: -17.2641735, 25.5135803, -19.8761311, 30.3878574, -47.6520309, 45.3897095
2: -17.7096806, 25.1806221, -20.3709641, 29.8667965, -47.5764732, 45.5515862
3: -21.0906239, 29.2755661, -24.4481449, 34.9785995, -56.0692177, 53.7237053
4: -19.5812111, 28.0415192, -22.9578114, 33.1388779, -52.7200890, 50.9993172

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6763389, upper bound: 52.4628825
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6195791, upper bound: 52.3065274
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19.5251865, 35.8567772, -19.0125427, 35.1322556, -54.6574364, 54.8693161
1: -21.9943600, 33.1002808, -21.4134941, 32.7879028, -54.7822647, 54.5137672
2: -22.4911900, 32.4780884, -21.9236660, 32.1679916, -54.6591759, 54.4017487
3: -27.0581684, 38.1643448, -26.3699760, 37.8068047, -64.8649750, 64.5343094
4: -25.2481842, 36.2807236, -24.7712917, 35.7631111, -61.0112953, 61.0520020

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8583024, upper bound: 52.8539414
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8214052, upper bound: 52.6903222
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.3688536, 28.6628151, -20.3451557, 37.2922020, -52.6610565, 49.0079727
1: -17.2641735, 25.5135803, -22.8790359, 34.4719276, -51.7360992, 48.3926125
2: -17.7096806, 25.1806221, -23.4392948, 33.8204613, -51.5301361, 48.6199188
3: -21.0906239, 29.2755661, -28.1053257, 39.7860641, -60.8766861, 57.3808899
4: -19.5812111, 28.0415192, -26.3313980, 37.7500992, -57.3313026, 54.3729095

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5045472, upper bound: 52.5045472
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5045472, upper bound: 53.0435062
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19.5251865, 35.8567772, -21.6481285, 39.5172691, -59.0424576, 57.5049019
1: -21.9943600, 33.1002808, -24.3567982, 36.7261505, -58.7205124, 57.4570732
2: -22.4911900, 32.4780884, -24.9315796, 35.9881516, -58.4793358, 57.4096603
3: -27.0581684, 38.1643448, -29.9461422, 42.4385223, -69.4966888, 68.1104813
4: -25.2481842, 36.2807236, -28.0547447, 40.2319031, -65.4800873, 64.3354645

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8475970, upper bound: 52.8725493
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8475970, upper bound: 53.5258721
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.6254129, 24.7713547, -23.5083370, 43.5609932, -56.1864052, 48.2796898
1: -14.2655630, 22.4507809, -26.5329418, 40.2712250, -54.5367889, 48.9837227
2: -14.6793137, 22.1696510, -27.0840607, 39.4476700, -54.1269836, 49.2537079
3: -17.5229568, 25.6474609, -32.7493973, 46.5774879, -64.1004486, 58.3968582
4: -16.6304893, 24.3550930, -30.5417747, 44.2039719, -60.8344612, 54.8968620

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8743135, upper bound: 53.8466913
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8743135, upper bound: 53.8466913
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.6885767, 26.5523262, -22.6732311, 42.0803375, -55.7689133, 49.2255554
1: -15.4345446, 24.2497425, -25.5944099, 38.8738441, -54.3083839, 49.8441544
2: -15.8949060, 23.9035759, -26.1259537, 38.0925484, -53.9874535, 50.0295258
3: -18.8986511, 27.7512016, -31.5871849, 44.9518166, -63.8504677, 59.3383865
4: -18.0523510, 26.2602692, -29.4729195, 42.6573067, -60.7096558, 55.7331886

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8743135, upper bound: 53.8466913
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8743135, upper bound: 54.1000274
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.6254129, 24.7713547, -25.8506546, 47.4542923, -60.0797043, 50.6219978
1: -14.2655630, 22.4507809, -29.1244144, 43.7585106, -58.0240746, 51.5751915
2: -14.6793137, 22.1696510, -29.7637978, 42.8362465, -57.5155602, 51.9334412
3: -17.5229568, 25.6474609, -35.8765335, 50.7101250, -68.2330780, 61.5239944
4: -16.6304893, 24.3550930, -33.4723434, 48.1443787, -64.7748718, 57.8274384

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5217012, upper bound: 53.6104243
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9663191, upper bound: 54.1352615
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8547619, upper bound: 53.8245792
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.6885767, 26.5523262, -25.0242729, 45.9716415, -59.6602135, 51.5765991
1: -15.4345446, 24.2497425, -28.1964779, 42.3377914, -57.7723274, 52.4462204
2: -15.8949060, 23.9035759, -28.8109703, 41.4669037, -57.3618088, 52.7145348
3: -18.8986511, 27.7512016, -34.7280350, 49.0487976, -67.9474487, 62.4792175
4: -18.0523510, 26.2602692, -32.3919678, 46.5843239, -64.6366730, 58.6522369

Time for backsubstitution: 0.90 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.25 + 418.56 = 420.81 seconds
