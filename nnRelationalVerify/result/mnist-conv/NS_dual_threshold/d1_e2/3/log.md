## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.061818120000000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.2950010, -10.6801090, -11.2950010, -10.6801090, -0.2605984, 0.2605989)
1: (-3.8135042, -3.4477706, -3.8135042, -3.4477706, -0.1932678, 0.1932678)
2: (-8.0900307, -7.6165128, -8.0900307, -7.6165128, -0.2409225, 0.2409225)
3: (-1.3556256, -0.8649361, -1.3556256, -0.8649361, -0.2066152, 0.2066151)
4: (-4.5547295, -4.0992761, -4.5547295, -4.0992761, -0.1671135, 0.1671135)
5: (-6.9657192, -6.3892069, -6.9657192, -6.3892069, -0.1912547, 0.1912547)
6: (-15.2578030, -14.7664623, -15.2578030, -14.7664623, -0.2285557, 0.2285556)
7: (4.7426939, 4.9851351, 4.7426939, 4.9851351, -0.1057246, 0.1057247)
8: (-4.4065752, -3.8740158, -4.4065752, -3.8740158, -0.2120063, 0.2120063)
9: (-0.6118510, -0.2206483, -0.6118510, -0.2206483, -0.1830323, 0.1830324)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.76 + 33.76 = 56.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0686868, upper bound: 0.0686868

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2341
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 2341

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679458, upper bound: 0.0671615
time: 3.17 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675641, upper bound: 0.0675641
time: 3.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.79 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.79
Output dim: 7, lower bound: -0.0679458, upper bound: 0.0671615
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.79
Output dim: 7, lower bound: -0.0675641, upper bound: 0.0675641

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -11.2899504, -10.6793699, -11.2932262, -10.6801090, -0.2544975, 0.2558105
1: -3.8134737, -3.4490786, -3.8135006, -3.4483047, -0.1939292, 0.1923927
2: -8.0876808, -7.6164370, -8.0891762, -7.6165128, -0.2394738, 0.2418201
3: -1.3603551, -0.8717122, -1.3556256, -0.8673749, -0.1997615, 0.1952888
4: -4.5531220, -4.0967417, -4.5540991, -4.0992780, -0.1635635, 0.1645753
5: -6.9657640, -6.3892331, -6.9657207, -6.3892179, -0.1912707, 0.1912215
6: -15.2580242, -14.7671051, -15.2578001, -14.7666883, -0.2287018, 0.2279688
7: 4.7445955, 4.9871693, 4.7433619, 4.9851346, -0.1021991, 0.1028863
8: -4.4001079, -3.8724222, -4.4039679, -3.8740168, -0.2042426, 0.2070144
9: -0.6109223, -0.2270219, -0.6118510, -0.2228899, -0.1768410, 0.1748140

Time for backsubstitution: 8.30 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0664412, upper bound: 0.0665275
time: 3.82 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667756, upper bound: 0.0659912
time: 3.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.2933979, -10.6801128, -11.2946491, -10.6801071, -0.2533314, 0.2604477
1: -3.8135006, -3.4481215, -3.8135033, -3.4478292, -0.1931491, 0.1935070
2: -8.0894346, -7.6165128, -8.0899315, -7.6165128, -0.2412188, 0.2407165
3: -1.3556256, -0.8687484, -1.3556256, -0.8655753, -0.2066035, 0.1930382
4: -4.5531354, -4.0992775, -4.5544629, -4.0992765, -0.1617600, 0.1670324
5: -6.9657187, -6.3892365, -6.9657202, -6.3892136, -0.1912508, 0.1912367
6: -15.2577982, -14.7667198, -15.2578001, -14.7665100, -0.2285128, 0.2283849
7: 4.7441111, 4.9851336, 4.7429314, 4.9851341, -0.1003100, 0.1056904
8: -4.4044700, -3.8740196, -4.4062233, -3.8740177, -0.2036388, 0.2118353
9: -0.6118510, -0.2216692, -0.6118510, -0.2208533, -0.1830226, 0.1732460

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 2327

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669296, upper bound: 0.0660589
time: 3.12 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663939, upper bound: 0.0663938
time: 3.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.53 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 15.53
Output dim: 7, lower bound: -0.0664412, upper bound: 0.0665275
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 15.53
Output dim: 7, lower bound: -0.0667756, upper bound: 0.0659912
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.53
Output dim: 7, lower bound: -0.0669296, upper bound: 0.0660589
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.53
Output dim: 7, lower bound: -0.0663939, upper bound: 0.0663938

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -11.2818623, -10.6666279, -11.2903996, -10.6801090, -0.2356920, 0.2415893
1: -3.8104389, -3.4478049, -3.8123355, -3.4483047, -0.1905677, 0.1903138
2: -8.0821953, -7.6121874, -8.0872583, -7.6165128, -0.2306826, 0.2359469
3: -1.3607121, -0.8780825, -1.3556256, -0.8696001, -0.2004266, 0.1923903
4: -4.5467548, -4.0973997, -4.5518732, -4.0992775, -0.1588218, 0.1626241
5: -6.9603901, -6.3882837, -6.9638419, -6.3892179, -0.1833081, 0.1860020
6: -15.2508430, -14.7614822, -15.2552118, -14.7666883, -0.2150674, 0.2204591
7: 4.7420530, 4.9853730, 4.7433619, 4.9845047, -0.0997131, 0.0986814
8: -4.4031811, -3.8807449, -4.4039679, -3.8777218, -0.1959597, 0.1951808
9: -0.6062653, -0.2252083, -0.6101692, -0.2228889, -0.1745386, 0.1756290

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 326

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0657962, upper bound: 0.0657641
time: 3.14 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656949, upper bound: 0.0657658
time: 3.26 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -11.2829990, -10.6793709, -11.2924013, -10.6801090, -0.2269299, 0.2556973
1: -3.8123550, -3.4490786, -3.8133733, -3.4483047, -0.1903837, 0.1923444
2: -8.0849724, -7.6164370, -8.0888662, -7.6165128, -0.2281919, 0.2417654
3: -1.3603551, -0.8729897, -1.3556256, -0.8675206, -0.1995728, 0.1954702
4: -4.5522470, -4.0967417, -4.5539980, -4.0992775, -0.1604333, 0.1644546
5: -6.9644113, -6.3892331, -6.9655652, -6.3892179, -0.1817189, 0.1912086
6: -15.2551336, -14.7671051, -15.2574453, -14.7666883, -0.2125331, 0.2278224
7: 4.7445955, 4.9857903, 4.7433619, 4.9849758, -0.1021689, 0.0970669
8: -4.4001079, -3.8747621, -4.4039679, -3.8742857, -0.2042289, 0.1934475
9: -0.6091764, -0.2270219, -0.6116266, -0.2228889, -0.1771941, 0.1745646

Time for backsubstitution: 8.47 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660214, upper bound: 0.0653461
time: 3.21 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660214, upper bound: 0.0652371
time: 3.12 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.2905693, -10.6801109, -11.2865591, -10.6673679, -0.2391090, 0.2416427
1: -3.8123336, -3.4481215, -3.8104696, -3.4465532, -0.1910706, 0.1901461
2: -8.0875149, -7.6165128, -8.0844421, -7.6122637, -0.2353454, 0.2319256
3: -1.3556256, -0.8709745, -1.3559816, -0.8719437, -0.2036922, 0.1937033
4: -4.5509100, -4.0992780, -4.5480971, -4.0999355, -0.1598088, 0.1622906
5: -6.9638414, -6.3892365, -6.9603453, -6.3882618, -0.1860312, 0.1832739
6: -15.2552090, -14.7667198, -15.2506199, -14.7608843, -0.2210028, 0.2147514
7: 4.7441111, 4.9845047, 4.7403889, 4.9833384, -0.0961049, 0.1032044
8: -4.4044700, -3.8777237, -4.4092979, -3.8823395, -0.1918049, 0.2035526
9: -0.6101692, -0.2216680, -0.6071935, -0.2190421, -0.1838571, 0.1709435

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 326

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662846, upper bound: 0.0653125
time: 3.32 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0661680, upper bound: 0.0653126
time: 3.08 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.2925730, -10.6801109, -11.2876968, -10.6801090, -0.2532187, 0.2328808
1: -3.8133717, -3.4481215, -3.8123868, -3.4478292, -0.1931009, 0.1899618
2: -8.0891228, -7.6165128, -8.0872211, -7.6165128, -0.2411642, 0.2294345
3: -1.3556256, -0.8688955, -1.3556256, -0.8668509, -0.2067720, 0.1928498
4: -4.5530357, -4.0992780, -4.5535908, -4.0992765, -0.1616393, 0.1639023
5: -6.9655652, -6.3892365, -6.9643679, -6.3892136, -0.1912379, 0.1816850
6: -15.2574444, -14.7667198, -15.2549105, -14.7665100, -0.2283664, 0.2122173
7: 4.7441111, 4.9849753, 4.7429314, 4.9837551, -0.0944904, 0.1056602
8: -4.4044700, -3.8742876, -4.4062233, -3.8763580, -0.1900716, 0.2118214
9: -0.6116266, -0.2216680, -0.6101029, -0.2208538, -0.1827742, 0.1735948

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 326
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 326

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0657488, upper bound: 0.0656397
time: 3.07 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656397, upper bound: 0.0656397
time: 2.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.37 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0657962, upper bound: 0.0657641
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0656949, upper bound: 0.0657658
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0660214, upper bound: 0.0653461
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0660214, upper bound: 0.0652371
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0662846, upper bound: 0.0653125
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0661680, upper bound: 0.0653126
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0657488, upper bound: 0.0656397
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.37
Output dim: 7, lower bound: -0.0656397, upper bound: 0.0656397

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -11.2811127, -10.6621590, -11.2901402, -10.6801109, -0.2304037, 0.2355881
1: -3.8096790, -3.4405575, -3.8120832, -3.4483047, -0.1849594, 0.1856456
2: -8.0817499, -7.6131973, -8.0872564, -7.6168966, -0.2295184, 0.2347724
3: -1.3598819, -0.8786130, -1.3553398, -0.8696008, -0.1995813, 0.1915456
4: -4.5443683, -4.1020560, -4.5518732, -4.1009083, -0.1526376, 0.1570027
5: -6.9573069, -6.3875785, -6.9627776, -6.3892179, -0.1775048, 0.1794643
6: -15.2488308, -14.7596493, -15.2544823, -14.7666883, -0.2113848, 0.2175560
7: 4.7434502, 4.9858150, 4.7438440, 4.9845047, -0.0969899, 0.0957096
8: -4.4073691, -3.8833356, -4.4039679, -3.8786983, -0.1883190, 0.1872483
9: -0.6054225, -0.2276046, -0.6101692, -0.2237234, -0.1714225, 0.1725948

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652096, upper bound: 0.0634375
time: 3.16 seconds

## Relational analysis of NS_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652097, upper bound: 0.0652181
time: 3.33 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -11.2774944, -10.6666317, -11.2889843, -10.6801100, -0.2237144, 0.2414732
1: -3.8040524, -3.4478049, -3.8102956, -3.4483047, -0.1799192, 0.1899064
2: -8.0821924, -7.6125078, -8.0872593, -7.6166224, -0.2305870, 0.2342117
3: -1.3605635, -0.8780820, -1.3555799, -0.8696024, -0.1999224, 0.1922768
4: -4.5467548, -4.0987043, -4.5518732, -4.0996966, -0.1586192, 0.1549699
5: -6.9575796, -6.3882837, -6.9629784, -6.3892179, -0.1731285, 0.1859865
6: -15.2480383, -14.7614822, -15.2543163, -14.7666883, -0.2094855, 0.2203722
7: 4.7434020, 4.9853735, 4.7437873, 4.9845052, -0.0949824, 0.0986274
8: -4.4031811, -3.8861713, -4.4039679, -3.8794241, -0.1955398, 0.1816422
9: -0.6062653, -0.2261908, -0.6101692, -0.2231915, -0.1744505, 0.1709528

Time for backsubstitution: 9.05 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A1_A1_A2_A1

### Relational analysis result of NS_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651088, upper bound: 0.0634393
time: 3.67 seconds

## Relational analysis of NS_A1_A1_A2_A2

### Relational analysis result of NS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0652193
time: 3.27 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -11.2827406, -10.6793690, -11.2916527, -10.6756439, -0.2202427, 0.2501466
1: -3.8120756, -3.4490786, -3.8126421, -3.4410539, -0.1857842, 0.1868894
2: -8.0849695, -7.6168180, -8.0884781, -7.6175518, -0.2270172, 0.2407470
3: -1.3600707, -0.8729892, -1.3547959, -0.8680520, -0.1993088, 0.1949104
4: -4.5522470, -4.0983734, -4.5517955, -4.1037393, -0.1559839, 0.1588631
5: -6.9633460, -6.3892331, -6.9624805, -6.3885136, -0.1751931, 0.1854334
6: -15.2543669, -14.7671051, -15.2554855, -14.7648392, -0.2095772, 0.2243032
7: 4.7450771, 4.9857903, 4.7447596, 4.9854064, -0.0993485, 0.0943747
8: -4.4001079, -3.8757906, -4.4081016, -3.8769560, -0.1976469, 0.1863623
9: -0.6091764, -0.2278574, -0.6108651, -0.2252851, -0.1741036, 0.1713377

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654764, upper bound: 0.0631051
time: 3.10 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654762, upper bound: 0.0648010
time: 2.88 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -11.2815838, -10.6793690, -11.2880325, -10.6801119, -0.2268128, 0.2441466
1: -3.8103848, -3.4490786, -3.8067999, -3.4483047, -0.1899753, 0.1817815
2: -8.0849714, -7.6165447, -8.0888643, -7.6168671, -0.2263331, 0.2416695
3: -1.3603110, -0.8729892, -1.3554790, -0.8675210, -0.1995361, 0.1946031
4: -4.5522470, -4.0971603, -4.5539980, -4.1006370, -0.1528330, 0.1644534
5: -6.9635458, -6.3892331, -6.9627538, -6.3892179, -0.1817036, 0.1810449
6: -15.2542715, -14.7671051, -15.2546053, -14.7666883, -0.2124350, 0.2223873
7: 4.7450194, 4.9857903, 4.7446899, 4.9849753, -0.1021147, 0.0922491
8: -4.4001079, -3.8764648, -4.4039679, -3.8798122, -0.1910664, 0.1933180
9: -0.6091764, -0.2273252, -0.6116266, -0.2238715, -0.1726478, 0.1744642

Time for backsubstitution: 8.68 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 311

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637809, upper bound: 0.0646925
time: 2.69 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654763, upper bound: 0.0646920
time: 2.79 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.2898178, -10.6756439, -11.2863045, -10.6673708, -0.2337654, 0.2349325
1: -3.8116031, -3.4408703, -3.8102174, -3.4465532, -0.1856136, 0.1854914
2: -8.0871334, -7.6175518, -8.0844431, -7.6126366, -0.2342939, 0.2307682
3: -1.3547959, -0.8715050, -1.3556943, -0.8719437, -0.2032098, 0.1928343
4: -4.5486417, -4.1037393, -4.5480971, -4.1015687, -0.1532600, 0.1580977
5: -6.9607577, -6.3885293, -6.9592810, -6.3882618, -0.1802502, 0.1767821
6: -15.2532425, -14.7648726, -15.2498903, -14.7608843, -0.2174778, 0.2118034
7: 4.7454700, 4.9849401, 4.7408729, 4.9833379, -0.0935014, 0.1003620
8: -4.4086852, -3.8803630, -4.4092979, -3.8832664, -0.1838174, 0.1969345
9: -0.6094067, -0.2240162, -0.6071935, -0.2198756, -0.1805139, 0.1681560

Time for backsubstitution: 8.10 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0657386, upper bound: 0.0633697
time: 2.81 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0657384, upper bound: 0.0647264
time: 2.81 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.2862186, -10.6801128, -11.2851439, -10.6673708, -0.2277818, 0.2416031
1: -3.8057590, -3.4481215, -3.8084283, -3.4465532, -0.1806118, 0.1897378
2: -8.0875149, -7.6168671, -8.0844440, -7.6123743, -0.2352788, 0.2300766
3: -1.3554790, -0.8709736, -1.3559361, -0.8719456, -0.2034266, 0.1936855
4: -4.5509100, -4.1006365, -4.5480971, -4.1003566, -0.1598083, 0.1563871
5: -6.9610291, -6.3892365, -6.9594822, -6.3882618, -0.1758382, 0.1832585
6: -15.2523727, -14.7667198, -15.2497215, -14.7608843, -0.2155668, 0.2146542
7: 4.7454429, 4.9845037, 4.7408142, 4.9833374, -0.0913848, 0.1031503
8: -4.4044700, -3.8832464, -4.4092979, -3.8840404, -0.1916753, 0.1916350
9: -0.6101692, -0.2226512, -0.6071935, -0.2193425, -0.1837573, 0.1667020

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656217, upper bound: 0.0633697
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656218, upper bound: 0.0647266
time: 2.75 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.2918215, -10.6756439, -11.2874384, -10.6801109, -0.2478743, 0.2261932
1: -3.8126402, -3.4408703, -3.8121338, -3.4478292, -0.1876464, 0.1852987
2: -8.0887327, -7.6175518, -8.0872202, -7.6168966, -0.2401507, 0.2282605
3: -1.3547959, -0.8694260, -1.3553398, -0.8668518, -0.2062119, 0.1920396
4: -4.5508366, -4.1037393, -4.5535908, -4.1009092, -0.1551648, 0.1594529
5: -6.9624810, -6.3885293, -6.9633074, -6.3892136, -0.1854625, 0.1751492
6: -15.2554836, -14.7648726, -15.2541838, -14.7665100, -0.2248478, 0.2092462
7: 4.7454700, 4.9854059, 4.7434130, 4.9837551, -0.0917660, 0.1028515
8: -4.4086852, -3.8769569, -4.4062233, -3.8773842, -0.1820168, 0.2052395
9: -0.6108651, -0.2240162, -0.6101029, -0.2216890, -0.1795456, 0.1705612

Time for backsubstitution: 8.07 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652038, upper bound: 0.0633987
time: 2.77 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0652038, upper bound: 0.0650946
time: 2.79 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.2882214, -10.6801128, -11.2862835, -10.6801090, -0.2418692, 0.2328420
1: -3.8067970, -3.4481215, -3.8103454, -3.4478292, -0.1826530, 0.1895536
2: -8.0891199, -7.6168671, -8.0872211, -7.6166224, -0.2410681, 0.2275689
3: -1.3554790, -0.8688958, -1.3555799, -0.8668518, -0.2064283, 0.1928128
4: -4.5530357, -4.1006365, -4.5535908, -4.0996957, -0.1616380, 0.1576181
5: -6.9627542, -6.3892365, -6.9635029, -6.3892136, -0.1810942, 0.1816696
6: -15.2546043, -14.7667198, -15.2540159, -14.7665100, -0.2229595, 0.2121201
7: 4.7454429, 4.9849749, 4.7433567, 4.9837551, -0.0896497, 0.1056061
8: -4.4044700, -3.8798141, -4.4062233, -3.8780575, -0.1899421, 0.2000071
9: -0.6116266, -0.2226517, -0.6101029, -0.2211554, -0.1826731, 0.1691073

Time for backsubstitution: 8.01 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650946, upper bound: 0.0633987
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650945, upper bound: 0.0650946
time: 2.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 13.58 seconds
NS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0652096, upper bound: 0.0634375
NS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0652097, upper bound: 0.0652181
NS_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0651088, upper bound: 0.0634393
NS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0652193
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0654764, upper bound: 0.0631051
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0654762, upper bound: 0.0648010
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0637809, upper bound: 0.0646925
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0654763, upper bound: 0.0646920
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0657386, upper bound: 0.0633697
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0657384, upper bound: 0.0647264
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0656217, upper bound: 0.0633697
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0656218, upper bound: 0.0647266
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0652038, upper bound: 0.0633987
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0652038, upper bound: 0.0650946
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0650946, upper bound: 0.0633987
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.58
Output dim: 7, lower bound: -0.0650945, upper bound: 0.0650946

## BFS NS instance: NS_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -11.2811127, -10.6626358, -11.2901402, -10.6802397, -0.2302079, 0.2351663
1: -3.8090372, -3.4405575, -3.8119206, -3.4483047, -0.1843796, 0.1855074
2: -8.0805387, -7.6131973, -8.0869513, -7.6168966, -0.2285051, 0.2346313
3: -1.3590167, -0.8786237, -1.3551219, -0.8696055, -0.1984202, 0.1911452
4: -4.5443683, -4.1033163, -4.5518732, -4.1012244, -0.1521616, 0.1553640
5: -6.9573030, -6.3894930, -6.9627786, -6.3897257, -0.1763089, 0.1750487
6: -15.2468939, -14.7596493, -15.2539968, -14.7666883, -0.2101674, 0.2172590
7: 4.7434502, 4.9843059, 4.7438440, 4.9841251, -0.0967889, 0.0949225
8: -4.4073691, -3.8843632, -4.4039679, -3.8789501, -0.1881757, 0.1865040
9: -0.5957940, -0.2276106, -0.6076858, -0.2237263, -0.1628064, 0.1702354

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0632677, upper bound: 0.0606552
time: 2.92 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619962, upper bound: 0.0598223
time: 2.97 seconds

## BFS NS instance: NS_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -11.2810535, -10.6634693, -11.2901402, -10.6805525, -0.2301114, 0.2346885
1: -3.8084011, -3.4402781, -3.8116283, -3.4483047, -0.1839719, 0.1849806
2: -8.0794697, -7.6127205, -8.0864429, -7.6168966, -0.2280195, 0.2339983
3: -1.3584290, -0.8781936, -1.3548360, -0.8696163, -0.1985137, 0.1919260
4: -4.5446105, -4.1044645, -4.5518732, -4.1017742, -0.1538070, 0.1550267
5: -6.9574594, -6.3922524, -6.9627752, -6.3908997, -0.1849340, 0.1743796
6: -15.2470770, -14.7573462, -15.2538242, -14.7666883, -0.2103090, 0.2164569
7: 4.7418761, 4.9844942, 4.7438440, 4.9840121, -0.0967383, 0.0953019
8: -4.4084783, -3.8844886, -4.4039679, -3.8791299, -0.1885878, 0.1865774
9: -0.6024048, -0.2138677, -0.6090517, -0.2237301, -0.1666143, 0.1863489

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630871, upper bound: 0.0618892
time: 3.02 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613632, upper bound: 0.0608043
time: 2.95 seconds

## BFS NS instance: NS_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -11.2774944, -10.6671066, -11.2889843, -10.6802387, -0.2235186, 0.2410517
1: -3.8034124, -3.4478049, -3.8101320, -3.4483047, -0.1793401, 0.1897677
2: -8.0809784, -7.6125078, -8.0869532, -7.6166224, -0.2295732, 0.2340698
3: -1.3596997, -0.8780947, -1.3553636, -0.8696046, -0.1987616, 0.1918762
4: -4.5467548, -4.0999641, -4.5518732, -4.1000137, -0.1581434, 0.1533312
5: -6.9575758, -6.3901987, -6.9629765, -6.3897257, -0.1719328, 0.1815713
6: -15.2461014, -14.7614822, -15.2538261, -14.7666883, -0.2082684, 0.2200757
7: 4.7434020, 4.9838619, 4.7437873, 4.9841251, -0.0947814, 0.0978402
8: -4.4031811, -3.8871994, -4.4039679, -3.8796725, -0.1953970, 0.1808980
9: -0.5966377, -0.2261958, -0.6076858, -0.2231927, -0.1658341, 0.1685933

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A2_A1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0631706, upper bound: 0.0606582
time: 2.77 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619061, upper bound: 0.0598273
time: 2.82 seconds

## BFS NS instance: NS_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -11.2774353, -10.6679363, -11.2889843, -10.6805525, -0.2234219, 0.2405744
1: -3.8027835, -3.4475250, -3.8098395, -3.4483047, -0.1789322, 0.1892415
2: -8.0799084, -7.6120315, -8.0864449, -7.6166224, -0.2290859, 0.2334368
3: -1.3591106, -0.8776631, -1.3550782, -0.8696151, -0.1988553, 0.1926572
4: -4.5469980, -4.1011133, -4.5518732, -4.1005616, -0.1597891, 0.1529939
5: -6.9577293, -6.3929582, -6.9629755, -6.3908997, -0.1805578, 0.1809022
6: -15.2462835, -14.7591810, -15.2536583, -14.7666883, -0.2084098, 0.2192762
7: 4.7418280, 4.9840508, 4.7437873, 4.9840131, -0.0947306, 0.0982193
8: -4.4042921, -3.8873248, -4.4039679, -3.8798542, -0.1958100, 0.1809714
9: -0.6032478, -0.2124541, -0.6090517, -0.2231977, -0.1696417, 0.1847067

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A2_A2_B1

### Relational analysis result of NS_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0629915, upper bound: 0.0618922
time: 2.78 seconds

## Relational analysis of NS_A1_A1_A2_A2_B2

### Relational analysis result of NS_A1_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612722, upper bound: 0.0608096
time: 2.83 seconds

## BFS NS instance: NS_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.2827406, -10.6798830, -11.2916527, -10.6757698, -0.2201362, 0.2493668
1: -3.8114834, -3.4490786, -3.8124876, -3.4410539, -0.1852082, 0.1867584
2: -8.0837555, -7.6168180, -8.0881701, -7.6175518, -0.2260027, 0.2406007
3: -1.3592045, -0.8730035, -1.3545775, -0.8680558, -0.1979750, 0.1945614
4: -4.5522470, -4.0996323, -4.5517955, -4.1040554, -0.1555339, 0.1570950
5: -6.9633427, -6.3911524, -6.9624796, -6.3890200, -0.1739625, 0.1811260
6: -15.2526340, -14.7671051, -15.2549982, -14.7648392, -0.2083615, 0.2239981
7: 4.7450771, 4.9842796, 4.7447596, 4.9850259, -0.0991473, 0.0935873
8: -4.4001079, -3.8767405, -4.4081016, -3.8771968, -0.1974874, 0.1856197
9: -0.5994182, -0.2278621, -0.6083779, -0.2252860, -0.1654875, 0.1692059

Time for backsubstitution: 8.76 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0639721, upper bound: 0.0611213
time: 2.81 seconds

## Relational analysis of NS_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0631459, upper bound: 0.0608422
time: 2.86 seconds

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.2827940, -10.6806145, -11.2916527, -10.6760864, -0.2198116, 0.2490788
1: -3.8107519, -3.4488864, -3.8121719, -3.4410539, -0.1846536, 0.1861267
2: -8.0826864, -7.6163425, -8.0876617, -7.6175518, -0.2255158, 0.2388713
3: -1.3587487, -0.8724892, -1.3542929, -0.8680665, -0.1980520, 0.1951793
4: -4.5524874, -4.1007814, -4.5517955, -4.1046047, -0.1569965, 0.1567602
5: -6.9634995, -6.3939075, -6.9624777, -6.3901944, -0.1827126, 0.1804304
6: -15.2525349, -14.7651129, -15.2548180, -14.7648392, -0.2083204, 0.2231312
7: 4.7435036, 4.9844155, 4.7447596, 4.9849133, -0.0984155, 0.0938655
8: -4.4010272, -3.8770676, -4.4081016, -3.8774137, -0.1966829, 0.1856030
9: -0.6063097, -0.2145412, -0.6097469, -0.2252903, -0.1692796, 0.1815172

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0637085, upper bound: 0.0622470
time: 2.98 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0622714, upper bound: 0.0615961
time: 2.95 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -11.2815838, -10.6794987, -11.2880325, -10.6806202, -0.2263938, 0.2439501
1: -3.8102379, -3.4490786, -3.8061824, -3.4483047, -0.1898303, 0.1812618
2: -8.0846643, -7.6165447, -8.0876493, -7.6168671, -0.2260416, 0.2410890
3: -1.3600943, -0.8729911, -1.3546140, -0.8675346, -0.1991394, 0.1934410
4: -4.5522470, -4.0974779, -4.5539980, -4.1018963, -0.1511944, 0.1639775
5: -6.9635444, -6.3897448, -6.9627485, -6.3911319, -0.1772877, 0.1798518
6: -15.2538338, -14.7671051, -15.2527876, -14.7666883, -0.2121296, 0.2211775
7: 4.7450194, 4.9854112, 4.7446899, 4.9834652, -0.1015142, 0.0919696
8: -4.4001079, -3.8767028, -4.4039679, -3.8807626, -0.1904941, 0.1931087
9: -0.6066909, -0.2273257, -0.6017711, -0.2238758, -0.1702365, 0.1668267

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0617966, upper bound: 0.0631874
time: 2.86 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0615181, upper bound: 0.0623619
time: 2.92 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -11.2815838, -10.6798134, -11.2879782, -10.6813526, -0.2259594, 0.2445886
1: -3.8099158, -3.4490786, -3.8054810, -3.4481120, -0.1893260, 0.1808090
2: -8.0841560, -7.6165447, -8.0865803, -7.6163898, -0.2257650, 0.2404813
3: -1.3598080, -0.8730040, -1.3541574, -0.8671012, -0.2002146, 0.1935991
4: -4.5522470, -4.0980277, -4.5542383, -4.1030445, -0.1510191, 0.1660875
5: -6.9635429, -6.3909154, -6.9629030, -6.3938928, -0.1766193, 0.1880306
6: -15.2535486, -14.7671051, -15.2528400, -14.7647018, -0.2113283, 0.2213163
7: 4.7450194, 4.9852991, 4.7431159, 4.9836006, -0.1018720, 0.0921515
8: -4.4001079, -3.8769188, -4.4048901, -3.8810806, -0.1905442, 0.1938769
9: -0.6080580, -0.2273297, -0.6087611, -0.2105556, -0.1866003, 0.1706449

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0629223, upper bound: 0.0629241
time: 2.80 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0622714, upper bound: 0.0614870
time: 2.83 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -11.2898178, -10.6761541, -11.2863045, -10.6674881, -0.2336584, 0.2341548
1: -3.8109603, -3.4408703, -3.8100550, -3.4465532, -0.1850595, 0.1853465
2: -8.0859203, -7.6175518, -8.0841389, -7.6126366, -0.2337317, 0.2304764
3: -1.3539300, -0.8715189, -1.3554754, -0.8719478, -0.2018762, 0.1924851
4: -4.5486417, -4.1049991, -4.5480971, -4.1018848, -0.1528099, 0.1563298
5: -6.9607563, -6.3904476, -6.9592814, -6.3887725, -0.1790214, 0.1724744
6: -15.2513046, -14.7648726, -15.2494049, -14.7608843, -0.2162974, 0.2114973
7: 4.7454700, 4.9834304, 4.7408729, 4.9829574, -0.0932219, 0.0996996
8: -4.4086852, -3.8813591, -4.4092979, -3.8835239, -0.1836079, 0.1964184
9: -0.5996537, -0.2240205, -0.6047645, -0.2198772, -0.1723762, 0.1657408

Time for backsubstitution: 8.83 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0633984, upper bound: 0.0609684
time: 2.84 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0620817, upper bound: 0.0602175
time: 2.81 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -11.2897606, -10.6768856, -11.2863045, -10.6678343, -0.2338918, 0.2338711
1: -3.8103228, -3.4406767, -3.8097627, -3.4465532, -0.1846323, 0.1847678
2: -8.0848503, -7.6170740, -8.0836296, -7.6126366, -0.2331142, 0.2292205
3: -1.3534744, -0.8710845, -1.3551722, -0.8719599, -0.2019553, 0.1932609
4: -4.5488806, -4.1061482, -4.5480971, -4.1024337, -0.1547607, 0.1559952
5: -6.9609084, -6.3932042, -6.9592795, -6.3899446, -0.1874595, 0.1717801
6: -15.2514887, -14.7628813, -15.2492352, -14.7608843, -0.2164309, 0.2106222
7: 4.7438955, 4.9835663, 4.7408729, 4.9828444, -0.0927714, 0.1000739
8: -4.4096045, -3.8815651, -4.4092979, -3.8836775, -0.1831116, 0.1964633
9: -0.6065418, -0.2107000, -0.6059473, -0.2198820, -0.1761900, 0.1789477

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0631308, upper bound: 0.0618878
time: 2.92 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613117, upper bound: 0.0608780
time: 2.91 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -11.2862186, -10.6806240, -11.2851439, -10.6674891, -0.2276746, 0.2408264
1: -3.8051167, -3.4481215, -3.8082671, -3.4465532, -0.1800582, 0.1895928
2: -8.0862999, -7.6168671, -8.0841379, -7.6123743, -0.2347159, 0.2297848
3: -1.3546140, -0.8709881, -1.3557186, -0.8719494, -0.2020931, 0.1933361
4: -4.5509100, -4.1018972, -4.5480971, -4.1006732, -0.1593582, 0.1546190
5: -6.9610271, -6.3911519, -6.9594808, -6.3887725, -0.1746093, 0.1789511
6: -15.2504358, -14.7667198, -15.2492361, -14.7608843, -0.2143860, 0.2143486
7: 4.7454429, 4.9829946, 4.7408142, 4.9829574, -0.0911053, 0.1024880
8: -4.4044700, -3.8842459, -4.4092979, -3.8842993, -0.1914657, 0.1911190
9: -0.6004167, -0.2226553, -0.6047645, -0.2193441, -0.1756192, 0.1642869

Time for backsubstitution: 8.84 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0632885, upper bound: 0.0609697
time: 2.71 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619736, upper bound: 0.0602213
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -11.2861614, -10.6813545, -11.2851439, -10.6678352, -0.2279075, 0.2405422
1: -3.8044865, -3.4479284, -3.8079739, -3.4465532, -0.1796308, 0.1890147
2: -8.0852289, -7.6163898, -8.0836296, -7.6123743, -0.2340975, 0.2285291
3: -1.3541574, -0.8705547, -1.3554144, -0.8719609, -0.2021720, 0.1941125
4: -4.5511503, -4.1030459, -4.5480971, -4.1012230, -0.1613092, 0.1542845
5: -6.9611816, -6.3939114, -6.9594793, -6.3899446, -0.1830477, 0.1782566
6: -15.2506180, -14.7647305, -15.2490654, -14.7608843, -0.2145199, 0.2134753
7: 4.7438703, 4.9831285, 4.7408142, 4.9828453, -0.0906552, 0.1028625
8: -4.4053893, -3.8844519, -4.4092979, -3.8844557, -0.1909698, 0.1911639
9: -0.6073052, -0.2093353, -0.6059473, -0.2193480, -0.1794337, 0.1774938

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0630211, upper bound: 0.0618893
time: 2.91 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612043, upper bound: 0.0608823
time: 2.75 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -11.2918215, -10.6761541, -11.2874384, -10.6802378, -0.2476778, 0.2257740
1: -3.8120227, -3.4408703, -3.8119850, -3.4478292, -0.1871264, 0.1851537
2: -8.0875187, -7.6175518, -8.0869141, -7.6168966, -0.2395711, 0.2279687
3: -1.3539300, -0.8694377, -1.3551219, -0.8668547, -0.2050507, 0.1916395
4: -4.5508366, -4.1049991, -4.5535908, -4.1012259, -0.1546887, 0.1578144
5: -6.9624772, -6.3904476, -6.9633036, -6.3897219, -0.1842692, 0.1707331
6: -15.2536592, -14.7648726, -15.2537441, -14.7665100, -0.2236376, 0.2089401
7: 4.7454700, 4.9838963, 4.7434130, 4.9833746, -0.0914866, 0.1022508
8: -4.4086852, -3.8779087, -4.4062233, -3.8776236, -0.1818072, 0.2046674
9: -0.6010087, -0.2240205, -0.6076174, -0.2216907, -0.1719089, 0.1681485

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0636996, upper bound: 0.0614148
time: 2.81 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0628738, upper bound: 0.0611359
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -11.2917633, -10.6768847, -11.2874384, -10.6805525, -0.2483163, 0.2253389
1: -3.8113165, -3.4406767, -3.8116617, -3.4478292, -0.1866739, 0.1846497
2: -8.0864525, -7.6170740, -8.0864048, -7.6168966, -0.2389655, 0.2276911
3: -1.3534744, -0.8690028, -1.3548360, -0.8668656, -0.2052140, 0.1927086
4: -4.5510745, -4.1061482, -4.5535908, -4.1017737, -0.1567987, 0.1576391
5: -6.9626341, -6.3932042, -6.9633021, -6.3908944, -0.1924479, 0.1700647
6: -15.2537098, -14.7628813, -15.2534599, -14.7665100, -0.2237766, 0.2081375
7: 4.7438955, 4.9840307, 4.7434130, 4.9832625, -0.0916684, 0.1026084
8: -4.4096045, -3.8782258, -4.4062233, -3.8778410, -0.1825752, 0.2047173
9: -0.6079984, -0.2107000, -0.6089864, -0.2216935, -0.1757284, 0.1845098

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0634360, upper bound: 0.0625405
time: 2.83 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619991, upper bound: 0.0618896
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -11.2882214, -10.6806231, -11.2862835, -10.6802397, -0.2416731, 0.2324233
1: -3.8061845, -3.4481215, -3.8101966, -3.4478292, -0.1821330, 0.1894088
2: -8.0879059, -7.6168671, -8.0869150, -7.6166224, -0.2404878, 0.2272775
3: -1.3546140, -0.8689084, -1.3553636, -0.8668542, -0.2052671, 0.1924131
4: -4.5530357, -4.1018953, -4.5535908, -4.1000137, -0.1611620, 0.1559795
5: -6.9627504, -6.3911519, -6.9635048, -6.3897219, -0.1799008, 0.1772537
6: -15.2527838, -14.7667198, -15.2535791, -14.7665100, -0.2217493, 0.2118144
7: 4.7454429, 4.9834647, 4.7433567, 4.9833755, -0.0893703, 0.1050055
8: -4.4044700, -3.8807659, -4.4062233, -3.8782997, -0.1897328, 0.1994350
9: -0.6017711, -0.2226555, -0.6076174, -0.2211576, -0.1750361, 0.1666944

Time for backsubstitution: 8.88 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0635904, upper bound: 0.0614148
time: 2.72 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0627646, upper bound: 0.0611359
time: 2.61 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -11.2881632, -10.6813545, -11.2862835, -10.6805477, -0.2423109, 0.2319887
1: -3.8054798, -3.4479284, -3.8098731, -3.4478292, -0.1816804, 0.1889043
2: -8.0868397, -7.6163898, -8.0864067, -7.6166224, -0.2398806, 0.2269995
3: -1.3541574, -0.8684726, -1.3550782, -0.8668661, -0.2054304, 0.1934820
4: -4.5532751, -4.1030459, -4.5535908, -4.1005635, -0.1632720, 0.1558043
5: -6.9629030, -6.3939114, -6.9635000, -6.3908944, -0.1880798, 0.1765853
6: -15.2528343, -14.7647305, -15.2532902, -14.7665100, -0.2218883, 0.2110132
7: 4.7438703, 4.9836006, 4.7433567, 4.9832625, -0.0895523, 0.1053634
8: -4.4053893, -3.8810792, -4.4062233, -3.8785172, -0.1905007, 0.1994849
9: -0.6087611, -0.2093353, -0.6089864, -0.2211623, -0.1788563, 0.1830552

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 557
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0633268, upper bound: 0.0625405
time: 2.77 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0618899, upper bound: 0.0618897
time: 2.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 14.44 seconds
NS_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0632677, upper bound: 0.0606552
NS_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0619962, upper bound: 0.0598223
NS_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0630871, upper bound: 0.0618892
NS_A1_A1_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0613632, upper bound: 0.0608043
NS_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0631706, upper bound: 0.0606582
NS_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0619061, upper bound: 0.0598273
NS_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0629915, upper bound: 0.0618922
NS_A1_A1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0612722, upper bound: 0.0608096
NS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0639721, upper bound: 0.0611213
NS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0631459, upper bound: 0.0608422
NS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0637085, upper bound: 0.0622470
NS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0622714, upper bound: 0.0615961
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0617966, upper bound: 0.0631874
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0615181, upper bound: 0.0623619
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0629223, upper bound: 0.0629241
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0622714, upper bound: 0.0614870
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0633984, upper bound: 0.0609684
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0620817, upper bound: 0.0602175
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0631308, upper bound: 0.0618878
NS_A2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0613117, upper bound: 0.0608780
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0632885, upper bound: 0.0609697
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0619736, upper bound: 0.0602213
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0630211, upper bound: 0.0618893
NS_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0612043, upper bound: 0.0608823
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0636996, upper bound: 0.0614148
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0628738, upper bound: 0.0611359
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0634360, upper bound: 0.0625405
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0619991, upper bound: 0.0618896
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0635904, upper bound: 0.0614148
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0627646, upper bound: 0.0611359
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0633268, upper bound: 0.0625405
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.44
Output dim: 7, lower bound: -0.0618899, upper bound: 0.0618897

## BFS NS instance: NS_A1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -11.2811127, -10.6626358, -11.2901402, -10.6804676, -0.2293906, 0.2344685
1: -3.8090372, -3.4405575, -3.8112960, -3.4483047, -0.1837363, 0.1842164
2: -8.0805387, -7.6131973, -8.0846510, -7.6168966, -0.2285051, 0.2320127
3: -1.3590167, -0.8786237, -1.3551219, -0.8696828, -0.1981782, 0.1909717
4: -4.5443683, -4.1033163, -4.5514235, -4.1012292, -0.1521418, 0.1548312
5: -6.9573030, -6.3894930, -6.9627614, -6.3902922, -0.1758248, 0.1749500
6: -15.2468939, -14.7596493, -15.2527161, -14.7666883, -0.2101674, 0.2158903
7: 4.7434502, 4.9843059, 4.7438440, 4.9834995, -0.0957235, 0.0949225
8: -4.4073691, -3.8843632, -4.4033532, -3.8790674, -0.1877377, 0.1851227
9: -0.5957940, -0.2276106, -0.6076858, -0.2244596, -0.1621591, 0.1702354

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619963, upper bound: 0.0598223
time: 2.94 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619963, upper bound: 0.0598222
time: 2.92 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -11.2811127, -10.6629877, -11.2930155, -10.6785355, -0.2293317, 0.2420483
1: -3.8077364, -3.4405575, -3.8106160, -3.4444723, -0.1970737, 0.1816106
2: -8.0774250, -7.6131973, -8.0791035, -7.5979114, -0.2560534, 0.2296487
3: -1.3590167, -0.8787429, -1.3563690, -0.8691337, -0.1983930, 0.1930876
4: -4.5434585, -4.1033206, -4.5480676, -4.0983057, -0.1555773, 0.1547862
5: -6.9572802, -6.3900042, -6.9674177, -6.3919516, -0.1777424, 0.1802549
6: -15.2447510, -14.7596493, -15.2495852, -14.7528343, -0.2231812, 0.2134043
7: 4.7434502, 4.9827709, 4.7396317, 4.9792309, -0.0947096, 0.1055976
8: -4.4059877, -3.8845282, -4.3988118, -3.8725328, -0.1984798, 0.1894290
9: -0.5957940, -0.2279887, -0.6137214, -0.2247837, -0.1628165, 0.1749625

Time for backsubstitution: 8.68 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619963, upper bound: 0.0598223
time: 2.99 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619963, upper bound: 0.0598223
time: 2.95 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -11.2810535, -10.6634693, -11.2901402, -10.6807804, -0.2292929, 0.2339907
1: -3.8084011, -3.4402781, -3.8110046, -3.4483047, -0.1833284, 0.1836971
2: -8.0794697, -7.6127205, -8.0841446, -7.6168966, -0.2280195, 0.2316623
3: -1.3584290, -0.8781936, -1.3548360, -0.8696949, -0.1982719, 0.1917529
4: -4.5446105, -4.1044645, -4.5514235, -4.1017766, -0.1537884, 0.1544938
5: -6.9574594, -6.3922524, -6.9627566, -6.3914642, -0.1844594, 0.1742811
6: -15.2470770, -14.7573462, -15.2526331, -14.7666883, -0.2103090, 0.2151220
7: 4.7418761, 4.9844942, 4.7438440, 4.9833870, -0.0957458, 0.0953019
8: -4.4084783, -3.8844886, -4.4033532, -3.8792486, -0.1881766, 0.1851962
9: -0.6024048, -0.2138677, -0.6090517, -0.2244642, -0.1659675, 0.1863489

Time for backsubstitution: 8.22 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A1_A2_B1_A1

### Relational analysis result of NS_A1_A1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613634, upper bound: 0.0608044
time: 3.03 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_A2

### Relational analysis result of NS_A1_A1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0613634, upper bound: 0.0608044
time: 3.08 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -11.2774944, -10.6671066, -11.2889843, -10.6804667, -0.2226995, 0.2403538
1: -3.8034124, -3.4478049, -3.8095078, -3.4483047, -0.1786966, 0.1884776
2: -8.0809784, -7.6125078, -8.0846510, -7.6166224, -0.2295732, 0.2314491
3: -1.3596997, -0.8780947, -1.3553636, -0.8696833, -0.1985215, 0.1917030
4: -4.5467548, -4.0999641, -4.5514235, -4.1000175, -0.1581236, 0.1527984
5: -6.9575758, -6.3901987, -6.9629598, -6.3902922, -0.1714485, 0.1814725
6: -15.2461014, -14.7614822, -15.2525520, -14.7666883, -0.2082684, 0.2187228
7: 4.7434020, 4.9838619, 4.7437873, 4.9834995, -0.0937154, 0.0978402
8: -4.4031811, -3.8871994, -4.4033532, -3.8797908, -0.1949630, 0.1795169
9: -0.5966377, -0.2261958, -0.6076858, -0.2239273, -0.1651846, 0.1685933

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 962
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A2_A1_B1_A1

### Relational analysis result of NS_A1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619062, upper bound: 0.0598272
time: 2.87 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_A2

### Relational analysis result of NS_A1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619062, upper bound: 0.0598272
time: 2.83 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -11.2774944, -10.6674576, -11.2918606, -10.6785345, -0.2226379, 0.2479365
1: -3.8021474, -3.4478049, -3.8089736, -3.4444723, -0.1920359, 0.1858819
2: -8.0778627, -7.6125078, -8.0791044, -7.5976362, -0.2571135, 0.2291436
3: -1.3596997, -0.8782127, -1.3566115, -0.8691344, -0.1987312, 0.1938195
4: -4.5458460, -4.0999680, -4.5480676, -4.0970945, -0.1615610, 0.1527536
5: -6.9575524, -6.3907104, -6.9676161, -6.3919516, -0.1733664, 0.1867777
6: -15.2439642, -14.7614822, -15.2494793, -14.7528343, -0.2212832, 0.2164299
7: 4.7434020, 4.9823284, 4.7395735, 4.9792309, -0.0926936, 0.1085136
8: -4.4018021, -3.8873692, -4.3988118, -3.8732476, -0.2057525, 0.1838238
9: -0.5966377, -0.2265732, -0.6137214, -0.2242508, -0.1658168, 0.1733215

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A2_A1_B2_A1

### Relational analysis result of NS_A1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619062, upper bound: 0.0598272
time: 2.89 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_A2

### Relational analysis result of NS_A1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619062, upper bound: 0.0598272
time: 2.86 seconds

## BFS NS instance: NS_A1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -11.2774353, -10.6679363, -11.2889843, -10.6807785, -0.2226022, 0.2398767
1: -3.8027835, -3.4475250, -3.8092158, -3.4483047, -0.1782889, 0.1879587
2: -8.0799084, -7.6120315, -8.0841436, -7.6166224, -0.2290859, 0.2310982
3: -1.3591106, -0.8776631, -1.3550782, -0.8696949, -0.1986154, 0.1924843
4: -4.5469980, -4.1011133, -4.5514235, -4.1005659, -0.1597704, 0.1524611
5: -6.9577293, -6.3929582, -6.9629579, -6.3914642, -0.1800835, 0.1808037
6: -15.2462835, -14.7591810, -15.2524681, -14.7666883, -0.2084098, 0.2179567
7: 4.7418280, 4.9840508, 4.7437873, 4.9833884, -0.0937375, 0.0982193
8: -4.4042921, -3.8873248, -4.4033532, -3.8799725, -0.1954024, 0.1795902
9: -0.6032478, -0.2124541, -0.6090517, -0.2239318, -0.1689924, 0.1847067

Time for backsubstitution: 8.80 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 557
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 962
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: A, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: B, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_A1_A2_A2_B1_A1

### Relational analysis result of NS_A1_A1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612722, upper bound: 0.0608095
time: 3.03 seconds

## Relational analysis of NS_A1_A1_A2_A2_B1_A2

### Relational analysis result of NS_A1_A1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612722, upper bound: 0.0608095
time: 3.02 seconds

## BFS NS instance: NS_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.2827406, -10.6798830, -11.2916527, -10.6760006, -0.2191684, 0.2486591
1: -3.8114834, -3.4490786, -3.8118610, -3.4410539, -0.1845602, 0.1854408
2: -8.0837555, -7.6168180, -8.0858707, -7.6175518, -0.2260027, 0.2379661
3: -1.3592045, -0.8730035, -1.3545775, -0.8681340, -0.1977361, 0.1943718
4: -4.5522470, -4.0996323, -4.5513458, -4.1040597, -0.1555127, 0.1565621
5: -6.9633427, -6.3911524, -6.9624643, -6.3895879, -0.1734661, 0.1810245
6: -15.2526340, -14.7671051, -15.2537060, -14.7648392, -0.2083615, 0.2226228
7: 4.7450771, 4.9842796, 4.7447596, 4.9844003, -0.0980794, 0.0935873
8: -4.4001079, -3.8767405, -4.4074883, -3.8773136, -0.1970549, 0.1839867
9: -0.5994182, -0.2278621, -0.6083779, -0.2260213, -0.1647989, 0.1692059

Time for backsubstitution: 8.80 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.52 + 547.32 = 603.85 seconds
