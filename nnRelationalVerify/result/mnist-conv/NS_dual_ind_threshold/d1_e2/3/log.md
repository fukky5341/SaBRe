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
execution time: IAR + RelationalAnalysis = 22.05 + 33.39 = 55.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0686868, upper bound: 0.0686868

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2341
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2341

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0679458, upper bound: 0.0671615
time: 2.87 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0675641, upper bound: 0.0675641
time: 2.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.98 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.98
Output dim: 7, lower bound: -0.0679458, upper bound: 0.0671615
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.98
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

Time for backsubstitution: 8.66 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2327

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0673113, upper bound: 0.0656562
time: 3.16 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667755, upper bound: 0.0659912
time: 2.78 seconds

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

Time for backsubstitution: 8.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2327
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2327

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0669296, upper bound: 0.0660589
time: 2.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663939, upper bound: 0.0663938
time: 2.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.04 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 7, lower bound: -0.0673113, upper bound: 0.0656562
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 7, lower bound: -0.0667755, upper bound: 0.0659912
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 7, lower bound: -0.0669296, upper bound: 0.0660589
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 7, lower bound: -0.0663939, upper bound: 0.0663938

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -11.2871227, -10.6793709, -11.2851362, -10.6673708, -0.2402755, 0.2370050
1: -3.8123064, -3.4490786, -3.8104677, -3.4470291, -0.1918504, 0.1890315
2: -8.0857639, -7.6164370, -8.0836887, -7.6122637, -0.2336001, 0.2330295
3: -1.3603551, -0.8739388, -1.3559816, -0.8737445, -0.1968631, 0.1959345
4: -4.5508957, -4.0967417, -4.5477333, -4.0999365, -0.1616123, 0.1598336
5: -6.9638829, -6.3892331, -6.9603448, -6.3882675, -0.1860512, 0.1832586
6: -15.2554350, -14.7671051, -15.2506199, -14.7610683, -0.2211915, 0.2143352
7: 4.7445955, 4.9865408, 4.7408199, 4.9833379, -0.0979942, 0.1004001
8: -4.4001079, -3.8761272, -4.4070430, -3.8823423, -0.1924090, 0.1987319
9: -0.6092417, -0.2270217, -0.6071935, -0.2210770, -0.1776556, 0.1725252

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667654, upper bound: 0.0637214
time: 2.84 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0667654, upper bound: 0.0650697
time: 2.89 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -11.2891245, -10.6793699, -11.2862740, -10.6801119, -0.2543845, 0.2282441
1: -3.8133447, -3.4490786, -3.8123841, -3.4483047, -0.1938810, 0.1888466
2: -8.0873709, -7.6164370, -8.0864668, -7.6165128, -0.2394187, 0.2305384
3: -1.3603551, -0.8718596, -1.3556256, -0.8686507, -0.1999619, 0.1951001
4: -4.5530214, -4.0967417, -4.5532269, -4.0992775, -0.1634428, 0.1614453
5: -6.9656067, -6.3892331, -6.9643688, -6.3892179, -0.1912581, 0.1816695
6: -15.2576694, -14.7671051, -15.2549086, -14.7666883, -0.2285557, 0.2118007
7: 4.7445955, 4.9870110, 4.7433619, 4.9837546, -0.0963796, 0.1028559
8: -4.4001079, -3.8726940, -4.4039679, -3.8763571, -0.1906754, 0.2070007
9: -0.6106992, -0.2270219, -0.6101029, -0.2228904, -0.1765926, 0.1751764

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662303, upper bound: 0.0637501
time: 2.80 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0662305, upper bound: 0.0654461
time: 2.83 seconds

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

Time for backsubstitution: 8.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663836, upper bound: 0.0641238
time: 2.78 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0663836, upper bound: 0.0654724
time: 3.06 seconds

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

Time for backsubstitution: 8.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 311

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658488, upper bound: 0.0641529
time: 2.83 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0658488, upper bound: 0.0658488
time: 2.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.10 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0667654, upper bound: 0.0637214
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0667654, upper bound: 0.0650697
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0662303, upper bound: 0.0637501
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0662305, upper bound: 0.0654461
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0663836, upper bound: 0.0641238
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0663836, upper bound: 0.0654724
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0658488, upper bound: 0.0641529
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.10
Output dim: 7, lower bound: -0.0658488, upper bound: 0.0658488

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.2871227, -10.6798820, -11.2851362, -10.6674900, -0.2401693, 0.2362250
1: -3.8116658, -3.4490786, -3.8103051, -3.4470291, -0.1912997, 0.1888866
2: -8.0845499, -7.6164370, -8.0833845, -7.6122637, -0.2330375, 0.2327380
3: -1.3594928, -0.8739498, -1.3557637, -0.8737473, -0.1955291, 0.1955854
4: -4.5508957, -4.0980005, -4.5477333, -4.1002531, -0.1611622, 0.1580656
5: -6.9638824, -6.3911524, -6.9603453, -6.3887749, -0.1848228, 0.1789510
6: -15.2535028, -14.7671051, -15.2501345, -14.7610683, -0.2200142, 0.2140297
7: 4.7445955, 4.9850307, 4.7408199, 4.9829569, -0.0977150, 0.0997366
8: -4.4001079, -3.8771238, -4.4070430, -3.8825998, -0.1921998, 0.1982170
9: -0.5994899, -0.2270279, -0.6047645, -0.2210786, -0.1695176, 0.1701102

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660019, upper bound: 0.0630763
time: 3.02 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660037, upper bound: 0.0629675
time: 2.86 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.2870674, -10.6806145, -11.2851362, -10.6678371, -0.2404050, 0.2359370
1: -3.8110321, -3.4488864, -3.8100126, -3.4470291, -0.1908786, 0.1883085
2: -8.0834789, -7.6159592, -8.0828743, -7.6122637, -0.2324188, 0.2314827
3: -1.3590355, -0.8735194, -1.3554595, -0.8737588, -0.1956068, 0.1963618
4: -4.5511365, -4.0991497, -4.5477333, -4.1008019, -0.1631134, 0.1577311
5: -6.9640369, -6.3939075, -6.9603438, -6.3899498, -0.1932619, 0.1782566
6: -15.2536812, -14.7651129, -15.2499590, -14.7610683, -0.2201532, 0.2131563
7: 4.7430215, 4.9851665, 4.7408199, 4.9828444, -0.0972657, 0.1001089
8: -4.4010272, -3.8773336, -4.4070430, -3.8827562, -0.1917039, 0.1982636
9: -0.6063750, -0.2137065, -0.6059473, -0.2210822, -0.1733315, 0.1833177

Time for backsubstitution: 8.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660019, upper bound: 0.0644247
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0660037, upper bound: 0.0643239
time: 3.03 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.2891245, -10.6798830, -11.2862740, -10.6802387, -0.2541888, 0.2278223
1: -3.8127298, -3.4490786, -3.8122361, -3.4483047, -0.1933641, 0.1887019
2: -8.0861559, -7.6164370, -8.0861626, -7.6165128, -0.2388384, 0.2302470
3: -1.3594928, -0.8718717, -1.3554077, -0.8686554, -0.1988009, 0.1947002
4: -4.5530214, -4.0980005, -4.5532269, -4.0995932, -0.1629670, 0.1598068
5: -6.9656048, -6.3911524, -6.9643679, -6.3897257, -0.1900647, 0.1772536
6: -15.2558498, -14.7671051, -15.2544737, -14.7666883, -0.2273483, 0.2114955
7: 4.7445955, 4.9855013, 4.7433619, 4.9833746, -0.0961006, 0.1022539
8: -4.4001079, -3.8736448, -4.4039679, -3.8765965, -0.1904663, 0.2064296
9: -0.6008437, -0.2270272, -0.6076174, -0.2228920, -0.1689554, 0.1727643

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654763, upper bound: 0.0631051
time: 2.98 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654764, upper bound: 0.0629961
time: 2.88 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.2890701, -10.6806135, -11.2862740, -10.6805525, -0.2548294, 0.2273831
1: -3.8120277, -3.4488864, -3.8119111, -3.4483047, -0.1929176, 0.1881979
2: -8.0850878, -7.6159592, -8.0856524, -7.6165128, -0.2382312, 0.2299689
3: -1.3590355, -0.8714390, -1.3551221, -0.8686664, -0.1989619, 0.1957693
4: -4.5532618, -4.0991497, -4.5532269, -4.1001425, -0.1650771, 0.1596314
5: -6.9657598, -6.3939075, -6.9643660, -6.3908997, -0.1982441, 0.1765856
6: -15.2558994, -14.7651129, -15.2541847, -14.7666883, -0.2274930, 0.2106941
7: 4.7430215, 4.9856367, 4.7433619, 4.9832625, -0.0962834, 0.1026096
8: -4.4010272, -3.8739610, -4.4039679, -3.8768158, -0.1912347, 0.2064813
9: -0.6078336, -0.2137065, -0.6089864, -0.2228961, -0.1727749, 0.1891255

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654762, upper bound: 0.0648010
time: 3.03 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0654764, upper bound: 0.0646920
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.2905693, -10.6806221, -11.2865591, -10.6674881, -0.2390023, 0.2408655
1: -3.8116899, -3.4481215, -3.8103065, -3.4465532, -0.1905198, 0.1900008
2: -8.0862999, -7.6165128, -8.0841389, -7.6122637, -0.2347822, 0.2316337
3: -1.3547609, -0.8709881, -1.3557637, -0.8719471, -0.2023582, 0.1933542
4: -4.5509100, -4.1005363, -4.5480971, -4.1002531, -0.1593588, 0.1605226
5: -6.9638381, -6.3911519, -6.9603467, -6.3887725, -0.1848028, 0.1789664
6: -15.2532721, -14.7667198, -15.2501326, -14.7608843, -0.2198262, 0.2144458
7: 4.7441111, 4.9829931, 4.7403889, 4.9829569, -0.0958254, 0.1025419
8: -4.4044700, -3.8787203, -4.4092979, -3.8826008, -0.1915954, 0.2030382
9: -0.6004167, -0.2216749, -0.6047645, -0.2190437, -0.1757188, 0.1685287

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656201, upper bound: 0.0634786
time: 2.97 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656219, upper bound: 0.0633698
time: 3.18 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.2905130, -10.6813545, -11.2865591, -10.6678352, -0.2392359, 0.2405820
1: -3.8110528, -3.4479284, -3.8100126, -3.4465532, -0.1900983, 0.1894231
2: -8.0852318, -7.6160378, -8.0836306, -7.6122637, -0.2341640, 0.2303784
3: -1.3543046, -0.8705554, -1.3554595, -0.8719594, -0.2024373, 0.1941303
4: -4.5511503, -4.1016846, -4.5480971, -4.1008024, -0.1613097, 0.1601884
5: -6.9639950, -6.3939114, -6.9603438, -6.3899446, -0.1932420, 0.1782721
6: -15.2534533, -14.7647305, -15.2499619, -14.7608843, -0.2199667, 0.2135719
7: 4.7425375, 4.9831295, 4.7403889, 4.9828448, -0.0953751, 0.1029165
8: -4.4053893, -3.8789258, -4.4092979, -3.8827524, -0.1910995, 0.2030853
9: -0.6073052, -0.2083521, -0.6059473, -0.2190475, -0.1795329, 0.1817360

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656201, upper bound: 0.0648274
time: 3.05 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0656219, upper bound: 0.0647265
time: 3.20 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.2925730, -10.6806221, -11.2876968, -10.6802340, -0.2530224, 0.2324619
1: -3.8127551, -3.4481215, -3.8122363, -3.4478292, -0.1925838, 0.1898168
2: -8.0879107, -7.6165128, -8.0869160, -7.6165128, -0.2405841, 0.2291429
3: -1.3547609, -0.8689072, -1.3554077, -0.8668544, -0.2056110, 0.1924498
4: -4.5530357, -4.1005363, -4.5535908, -4.0995946, -0.1611636, 0.1622640
5: -6.9655609, -6.3911519, -6.9643669, -6.3897219, -0.1900449, 0.1772691
6: -15.2556210, -14.7667198, -15.2544727, -14.7665100, -0.2271600, 0.2119116
7: 4.7441111, 4.9834642, 4.7429314, 4.9833751, -0.0942110, 0.1050595
8: -4.4044700, -3.8752379, -4.4062233, -3.8765969, -0.1898621, 0.2112510
9: -0.6017711, -0.2216749, -0.6076174, -0.2208552, -0.1751362, 0.1711824

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650946, upper bound: 0.0635079
time: 2.89 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650944, upper bound: 0.0633985
time: 2.87 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.2925167, -10.6813555, -11.2876968, -10.6805515, -0.2536600, 0.2320275
1: -3.8120484, -3.4479284, -3.8119111, -3.4478292, -0.1921375, 0.1893127
2: -8.0868387, -7.6160378, -8.0864067, -7.6165128, -0.2399766, 0.2288649
3: -1.3543046, -0.8684738, -1.3551221, -0.8668656, -0.2057736, 0.1935188
4: -4.5532751, -4.1016836, -4.5535908, -4.1001430, -0.1632737, 0.1620888
5: -6.9657168, -6.3939114, -6.9643664, -6.3908944, -0.1982245, 0.1766008
6: -15.2556725, -14.7647305, -15.2541876, -14.7665100, -0.2273064, 0.2111100
7: 4.7425375, 4.9836006, 4.7429314, 4.9832630, -0.0943928, 0.1054175
8: -4.4053893, -3.8755522, -4.4062233, -3.8768148, -0.1906303, 0.2113036
9: -0.6087611, -0.2083523, -0.6089864, -0.2208593, -0.1789565, 0.1875443

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 326

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650626, upper bound: 0.0652038
time: 2.90 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0650946, upper bound: 0.0650946
time: 2.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 14.98 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0660019, upper bound: 0.0630763
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0660037, upper bound: 0.0629675
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0660019, upper bound: 0.0644247
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0660037, upper bound: 0.0643239
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0654763, upper bound: 0.0631051
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0654764, upper bound: 0.0629961
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0654762, upper bound: 0.0648010
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0654764, upper bound: 0.0646920
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0656201, upper bound: 0.0634786
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0656219, upper bound: 0.0633698
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0656201, upper bound: 0.0648274
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0656219, upper bound: 0.0647265
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0650946, upper bound: 0.0635079
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0650944, upper bound: 0.0633985
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0650626, upper bound: 0.0652038
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.98
Output dim: 7, lower bound: -0.0650946, upper bound: 0.0650946

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.2868643, -10.6798820, -11.2843895, -10.6630230, -0.2334801, 0.2307252
1: -3.8113832, -3.4490786, -3.8095751, -3.4397821, -0.1867003, 0.1834246
2: -8.0845461, -7.6168180, -8.0830135, -7.6132765, -0.2318628, 0.2315675
3: -1.3592045, -0.8739510, -1.3549347, -0.8742788, -0.1951287, 0.1950260
4: -4.5508957, -4.0996327, -4.5453477, -4.1049075, -0.1567128, 0.1522903
5: -6.9628177, -6.3911524, -6.9572611, -6.3880696, -0.1782970, 0.1731644
6: -15.2527332, -14.7671051, -15.2481565, -14.7592335, -0.2170326, 0.2104924
7: 4.7450771, 4.9850302, 4.7422171, 4.9833989, -0.0948055, 0.0970445
8: -4.4001079, -3.8780994, -4.4112911, -3.8851871, -0.1855676, 0.1911316
9: -0.5994899, -0.2278616, -0.6040008, -0.2234738, -0.1664268, 0.1667279

Time for backsubstitution: 8.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0628051, upper bound: 0.0611081
time: 3.06 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623596, upper bound: 0.0599229
time: 3.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.2857084, -10.6798820, -11.2807703, -10.6674938, -0.2400531, 0.2247226
1: -3.8096936, -3.4490786, -3.8037329, -3.4470291, -0.1908922, 0.1783172
2: -8.0845480, -7.6165447, -8.0833817, -7.6126175, -0.2313094, 0.2326424
3: -1.3594458, -0.8739531, -1.3556159, -0.8737481, -0.1954920, 0.1948208
4: -4.5508957, -4.0984216, -4.5477333, -4.1015577, -0.1537963, 0.1580642
5: -6.9630165, -6.3911524, -6.9575338, -6.3887749, -0.1848074, 0.1687765
6: -15.2526369, -14.7671051, -15.2472925, -14.7610683, -0.2199168, 0.2085764
7: 4.7450194, 4.9850307, 4.7421699, 4.9829555, -0.0976608, 0.0949742
8: -4.4001079, -3.8788261, -4.4070430, -3.8881264, -0.1789871, 0.1980873
9: -0.5994899, -0.2273304, -0.6047645, -0.2220597, -0.1650668, 0.1700091

Time for backsubstitution: 8.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0628072, upper bound: 0.0610060
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623627, upper bound: 0.0598262
time: 3.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.2868099, -10.6806145, -11.2843895, -10.6633673, -0.2337158, 0.2304372
1: -3.8107486, -3.4488864, -3.8092799, -3.4397821, -0.1862791, 0.1828436
2: -8.0834770, -7.6163425, -8.0825062, -7.6132765, -0.2312446, 0.2303113
3: -1.3587487, -0.8735197, -1.3546281, -0.8742902, -0.1952065, 0.1958021
4: -4.5511365, -4.1007814, -4.5453477, -4.1054573, -0.1586637, 0.1519555
5: -6.9629712, -6.3939075, -6.9572592, -6.3892422, -0.1867359, 0.1724698
6: -15.2529125, -14.7651129, -15.2479839, -14.7592335, -0.2171696, 0.2096162
7: 4.7435036, 4.9851651, 4.7422171, 4.9832869, -0.0943562, 0.0974168
8: -4.4010272, -3.8783092, -4.4112911, -3.8853407, -0.1850703, 0.1911781
9: -0.6063750, -0.2145412, -0.6051831, -0.2234793, -0.1702406, 0.1799355

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0626741, upper bound: 0.0623007
time: 3.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615904, upper bound: 0.0605756
time: 3.09 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.2856512, -10.6806126, -11.2807703, -10.6678352, -0.2402883, 0.2244343
1: -3.8090615, -3.4488864, -3.8034413, -3.4470291, -0.1904712, 0.1777360
2: -8.0834780, -7.6160684, -8.0828733, -7.6126175, -0.2306924, 0.2313874
3: -1.3589895, -0.8735182, -1.3553114, -0.8737602, -0.1955696, 0.1955968
4: -4.5511365, -4.0995703, -4.5477333, -4.1021061, -0.1557474, 0.1577296
5: -6.9631724, -6.3939075, -6.9575315, -6.3899498, -0.1932462, 0.1680818
6: -15.2528172, -14.7651129, -15.2471228, -14.7610683, -0.2200572, 0.2076995
7: 4.7434468, 4.9851665, 4.7421699, 4.9828444, -0.0972116, 0.0953469
8: -4.4010272, -3.8790312, -4.4070430, -3.8882771, -0.1784898, 0.1981338
9: -0.6063750, -0.2140083, -0.6059473, -0.2220640, -0.1688806, 0.1832168

Time for backsubstitution: 8.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0626763, upper bound: 0.0622066
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615936, upper bound: 0.0604871
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.2888689, -10.6798840, -11.2855244, -10.6757708, -0.2474774, 0.2223231
1: -3.8124475, -3.4490786, -3.8115044, -3.4410539, -0.1887727, 0.1832401
2: -8.0861549, -7.6168180, -8.0857658, -7.6175518, -0.2376804, 0.2290593
3: -1.3592045, -0.8718719, -1.3545775, -0.8691862, -0.1983756, 0.1942184
4: -4.5530214, -4.0996327, -4.5510874, -4.1040554, -0.1587738, 0.1540303
5: -6.9645414, -6.3911524, -6.9612851, -6.3890200, -0.1835827, 0.1714672
6: -15.2550840, -14.7671051, -15.2525139, -14.7648392, -0.2244000, 0.2079585
7: 4.7450771, 4.9855013, 4.7447596, 4.9837985, -0.0931910, 0.0996827
8: -4.4001079, -3.8746500, -4.4081016, -3.8792953, -0.1838340, 0.1994117
9: -0.6008437, -0.2278616, -0.6068559, -0.2252862, -0.1660953, 0.1693269

Time for backsubstitution: 8.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651088, upper bound: 0.0630781
time: 3.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651088, upper bound: 0.0631048
time: 3.27 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.2877131, -10.6798840, -11.2819071, -10.6802387, -0.2540722, 0.2163417
1: -3.8107584, -3.4490786, -3.8056626, -3.4483047, -0.1929572, 0.1781250
2: -8.0861540, -7.6165447, -8.0861607, -7.6168671, -0.2371488, 0.2301803
3: -1.3594458, -0.8718731, -1.3552606, -0.8686550, -0.1987828, 0.1940475
4: -4.5530214, -4.0984216, -4.5532269, -4.1009531, -0.1559300, 0.1598063
5: -6.9647398, -6.3911524, -6.9615555, -6.3897257, -0.1900493, 0.1670353
6: -15.2549877, -14.7671051, -15.2516356, -14.7666883, -0.2272506, 0.2060112
7: 4.7450194, 4.9855013, 4.7446899, 4.9833736, -0.0960464, 0.0976462
8: -4.4001079, -3.8753433, -4.4039679, -3.8821239, -0.1772425, 0.2063003
9: -0.6008437, -0.2273295, -0.6076174, -0.2238727, -0.1647958, 0.1726635

Time for backsubstitution: 8.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0629690
time: 3.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0629674
time: 3.07 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.2888126, -10.6806145, -11.2855244, -10.6760855, -0.2481179, 0.2218838
1: -3.8117414, -3.4488864, -3.8111811, -3.4410539, -0.1883264, 0.1827329
2: -8.0850849, -7.6163425, -8.0852566, -7.6175518, -0.2370734, 0.2287813
3: -1.3587487, -0.8714390, -1.3542929, -0.8691969, -0.1985368, 0.1952875
4: -4.5532618, -4.1007814, -4.5510874, -4.1046057, -0.1608838, 0.1538548
5: -6.9646940, -6.3939075, -6.9612832, -6.3901944, -0.1917624, 0.1707986
6: -15.2551308, -14.7651129, -15.2522297, -14.7648392, -0.2245426, 0.2071538
7: 4.7435036, 4.9856367, 4.7447596, 4.9836874, -0.0933738, 0.1000384
8: -4.4010272, -3.8749666, -4.4081016, -3.8795147, -0.1846013, 0.1994632
9: -0.6078336, -0.2145410, -0.6082239, -0.2252901, -0.1699147, 0.1856880

Time for backsubstitution: 8.82 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0648010
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0644246
time: 3.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.2876539, -10.6806126, -11.2819071, -10.6805515, -0.2547131, 0.2159022
1: -3.8100576, -3.4488864, -3.8053389, -3.4483047, -0.1925111, 0.1776173
2: -8.0850840, -7.6160684, -8.0856514, -7.6168671, -0.2365432, 0.2299018
3: -1.3589895, -0.8714397, -1.3549762, -0.8686676, -0.1989440, 0.1951165
4: -4.5532618, -4.0995703, -4.5532269, -4.1015019, -0.1580402, 0.1596310
5: -6.9648938, -6.3939075, -6.9615498, -6.3908997, -0.1982287, 0.1663668
6: -15.2550354, -14.7651129, -15.2513485, -14.7666883, -0.2273965, 0.2052066
7: 4.7434468, 4.9856372, 4.7446899, 4.9832616, -0.0962293, 0.0980021
8: -4.4010272, -3.8756609, -4.4039679, -3.8823404, -0.1780097, 0.2063519
9: -0.6078336, -0.2140083, -0.6089864, -0.2238779, -0.1686152, 0.1890254

Time for backsubstitution: 8.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0646920
time: 3.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0643239
time: 3.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.2903118, -10.6806250, -11.2858105, -10.6630230, -0.2329973, 0.2353597
1: -3.8114400, -3.4481215, -3.8095765, -3.4393053, -0.1859665, 0.1843923
2: -8.0862980, -7.6168966, -8.0837669, -7.6132765, -0.2336087, 0.2304630
3: -1.3544745, -0.8709884, -1.3549347, -0.8724775, -0.2020383, 0.1925081
4: -4.5509100, -4.1021686, -4.5457110, -4.1049070, -0.1537375, 0.1556548
5: -6.9627748, -6.3911519, -6.9572620, -6.3880649, -0.1782850, 0.1731635
6: -15.2525434, -14.7667198, -15.2481594, -14.7590551, -0.2169485, 0.2107654
7: 4.7445803, 4.9829931, 4.7417870, 4.9833999, -0.0928532, 0.0998908
8: -4.4044700, -3.8796959, -4.4135857, -3.8851862, -0.1836644, 0.1967438
9: -0.6004167, -0.2225077, -0.6040008, -0.2214396, -0.1726285, 0.1654297

Time for backsubstitution: 8.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0624143, upper bound: 0.0615043
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619690, upper bound: 0.0603175
time: 2.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.2891550, -10.6806211, -11.2821941, -10.6674891, -0.2388864, 0.2286727
1: -3.8096499, -3.4481215, -3.8037345, -3.4465532, -0.1901124, 0.1792887
2: -8.0863018, -7.6166224, -8.0841370, -7.6126175, -0.2330041, 0.2315379
3: -1.3547144, -0.8709886, -1.3556159, -0.8719492, -0.2022444, 0.1922532
4: -4.5509100, -4.1009564, -4.5480971, -4.1015587, -0.1508213, 0.1603203
5: -6.9629707, -6.3911519, -6.9575348, -6.3887725, -0.1847873, 0.1687775
6: -15.2523775, -14.7667198, -15.2472944, -14.7608843, -0.2197392, 0.2088149
7: 4.7445402, 4.9829941, 4.7417383, 4.9829569, -0.0957713, 0.0978830
8: -4.4044700, -3.8804202, -4.4092979, -3.8881249, -0.1770180, 0.2026185
9: -0.6004167, -0.2219763, -0.6047645, -0.2200243, -0.1709859, 0.1684281

Time for backsubstitution: 8.87 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0624178, upper bound: 0.0614021
time: 2.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0619737, upper bound: 0.0602213
time: 2.95 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.2902546, -10.6813545, -11.2858105, -10.6633663, -0.2332306, 0.2350762
1: -3.8108020, -3.4479284, -3.8092833, -3.4393053, -0.1855447, 0.1838112
2: -8.0852318, -7.6164212, -8.0832605, -7.6132765, -0.2329898, 0.2292073
3: -1.3540170, -0.8705549, -1.3546281, -0.8724916, -0.2021176, 0.1932840
4: -4.5511503, -4.1033158, -4.5457110, -4.1054573, -0.1556881, 0.1553202
5: -6.9629288, -6.3939114, -6.9572592, -6.3892393, -0.1867242, 0.1724688
6: -15.2527285, -14.7647305, -15.2479887, -14.7590551, -0.2170843, 0.2098886
7: 4.7430067, 4.9831290, 4.7417870, 4.9832869, -0.0924030, 0.1002653
8: -4.4053893, -3.8799043, -4.4135857, -3.8853397, -0.1831665, 0.1967897
9: -0.6073052, -0.2091881, -0.6051831, -0.2214434, -0.1764420, 0.1786370

Time for backsubstitution: 8.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0622835, upper bound: 0.0626961
time: 2.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0611983, upper bound: 0.0609697
time: 3.05 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.2890968, -10.6813545, -11.2821941, -10.6678343, -0.2391196, 0.2283890
1: -3.8090131, -3.4479284, -3.8034430, -3.4465532, -0.1896908, 0.1787076
2: -8.0852318, -7.6161461, -8.0836287, -7.6126175, -0.2323871, 0.2302825
3: -1.3542595, -0.8705540, -1.3553114, -0.8719597, -0.2023239, 0.1930294
4: -4.5511503, -4.1021056, -4.5480971, -4.1021051, -0.1527722, 0.1599859
5: -6.9631281, -6.3939114, -6.9575324, -6.3899446, -0.1932266, 0.1680827
6: -15.2525587, -14.7647305, -15.2471256, -14.7608843, -0.2198794, 0.2079387
7: 4.7429667, 4.9831295, 4.7417383, 4.9828444, -0.0953209, 0.0982574
8: -4.4053893, -3.8806272, -4.4092979, -3.8882809, -0.1765202, 0.2026653
9: -0.6073052, -0.2086565, -0.6059473, -0.2200282, -0.1748000, 0.1816355

Time for backsubstitution: 8.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 312

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0622870, upper bound: 0.0626017
time: 2.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0612043, upper bound: 0.0608822
time: 2.90 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.2923136, -10.6806221, -11.2869473, -10.6757689, -0.2469943, 0.2269568
1: -3.8125038, -3.4481215, -3.8115063, -3.4405770, -0.1880393, 0.1842083
2: -8.0879097, -7.6168966, -8.0865202, -7.6175518, -0.2394261, 0.2279551
3: -1.3544745, -0.8689084, -1.3545775, -0.8673847, -0.2052660, 0.1916814
4: -4.5530357, -4.1021681, -4.5514507, -4.1040564, -0.1557986, 0.1573948
5: -6.9644980, -6.3911519, -6.9612837, -6.3890152, -0.1835707, 0.1714662
6: -15.2548952, -14.7667198, -15.2525167, -14.7646589, -0.2243159, 0.2082312
7: 4.7445803, 4.9834652, 4.7443295, 4.9837995, -0.0912388, 0.1025288
8: -4.4044700, -3.8762465, -4.4103999, -3.8792963, -0.1819308, 0.2050242
9: -0.6017711, -0.2225080, -0.6068559, -0.2232506, -0.1722915, 0.1680286

Time for backsubstitution: 8.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0634801
time: 2.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0635074
time: 3.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.2911587, -10.6806211, -11.2833300, -10.6802368, -0.2529056, 0.2202909
1: -3.8107147, -3.4481215, -3.8056636, -3.4478292, -0.1921775, 0.1790966
2: -8.0879078, -7.6166224, -8.0869141, -7.6168671, -0.2388442, 0.2290757
3: -1.3547144, -0.8689086, -1.3552606, -0.8668559, -0.2055160, 0.1914613
4: -4.5530357, -4.1009564, -4.5535908, -4.1009521, -0.1529550, 0.1620622
5: -6.9646959, -6.3911519, -6.9615541, -6.3897219, -0.1900294, 0.1670363
6: -15.2547264, -14.7667198, -15.2516375, -14.7665100, -0.2270730, 0.2062498
7: 4.7445402, 4.9834657, 4.7442603, 4.9833751, -0.0941569, 0.1005552
8: -4.4044700, -3.8769388, -4.4062233, -3.8821230, -0.1752735, 0.2108310
9: -0.6017711, -0.2219763, -0.6076174, -0.2218375, -0.1707091, 0.1710820

Time for backsubstitution: 8.84 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0633712
time: 3.08 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0633982
time: 2.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.2922573, -10.6813536, -11.2869473, -10.6760836, -0.2476323, 0.2265221
1: -3.8117971, -3.4479284, -3.8111835, -3.4405770, -0.1875925, 0.1837009
2: -8.0868378, -7.6164212, -8.0860100, -7.6175518, -0.2388191, 0.2276773
3: -1.3540170, -0.8684728, -1.3542929, -0.8673961, -0.2054288, 0.1927503
4: -4.5532751, -4.1033149, -4.5514507, -4.1046047, -0.1579083, 0.1572196
5: -6.9646525, -6.3939114, -6.9612818, -6.3901901, -0.1917503, 0.1707975
6: -15.2549467, -14.7647305, -15.2522278, -14.7646589, -0.2244577, 0.2074263
7: 4.7430067, 4.9836011, 4.7443295, 4.9836884, -0.0914207, 0.1028867
8: -4.4053893, -3.8765631, -4.4103999, -3.8795156, -0.1826974, 0.2050755
9: -0.6087611, -0.2091880, -0.6082239, -0.2232540, -0.1761110, 0.1843903

Time for backsubstitution: 8.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647271, upper bound: 0.0652039
time: 3.04 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647271, upper bound: 0.0648275
time: 4.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.2911015, -10.6813545, -11.2833300, -10.6805534, -0.2535439, 0.2198555
1: -3.8100090, -3.4479284, -3.8053420, -3.4478292, -0.1917310, 0.1785893
2: -8.0868387, -7.6161461, -8.0864058, -7.6168671, -0.2382383, 0.2287980
3: -1.3542595, -0.8684735, -1.3549762, -0.8668675, -0.2056787, 0.1925304
4: -4.5532751, -4.1021051, -4.5535908, -4.1015034, -0.1550649, 0.1618872
5: -6.9648509, -6.3939114, -6.9615541, -6.3908944, -0.1982090, 0.1663676
6: -15.2547760, -14.7647305, -15.2513514, -14.7665100, -0.2272193, 0.2054452
7: 4.7429667, 4.9836006, 4.7442603, 4.9832621, -0.0943386, 0.1009128
8: -4.4053893, -3.8772554, -4.4062233, -3.8823385, -0.1760399, 0.2108834
9: -0.6087611, -0.2086568, -0.6089864, -0.2218413, -0.1745285, 0.1874439

Time for backsubstitution: 8.87 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2327
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 557
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 962
type: A, layer: 3, pos: 558
type: A, layer: 3, pos: 1852
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1955
type: A, layer: 3, pos: 2568

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2327

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647270, upper bound: 0.0650946
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0647270, upper bound: 0.0647266
time: 2.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.02 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0628051, upper bound: 0.0611081
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0623596, upper bound: 0.0599229
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0628072, upper bound: 0.0610060
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0623627, upper bound: 0.0598262
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0626741, upper bound: 0.0623007
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0615904, upper bound: 0.0605756
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0626763, upper bound: 0.0622066
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0615936, upper bound: 0.0604871
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651088, upper bound: 0.0630781
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651088, upper bound: 0.0631048
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0629690
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0629674
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0648010
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0644246
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0646920
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0651087, upper bound: 0.0643239
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0624143, upper bound: 0.0615043
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0619690, upper bound: 0.0603175
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0624178, upper bound: 0.0614021
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0619737, upper bound: 0.0602213
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0622835, upper bound: 0.0626961
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0611983, upper bound: 0.0609697
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0622870, upper bound: 0.0626017
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0612043, upper bound: 0.0608822
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0634801
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0635074
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0633712
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647269, upper bound: 0.0633982
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647271, upper bound: 0.0652039
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647271, upper bound: 0.0648275
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647270, upper bound: 0.0650946
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -0.0647270, upper bound: 0.0647266

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.2868643, -10.6801167, -11.2843895, -10.6630230, -0.2327821, 0.2298919
1: -3.8107586, -3.4490786, -3.8095751, -3.4397821, -0.1854053, 0.1827816
2: -8.0822477, -7.6168180, -8.0830135, -7.6132765, -0.2292206, 0.2315675
3: -1.3592045, -0.8740332, -1.3549347, -0.8742788, -0.1949425, 0.1947635
4: -4.5504456, -4.0996351, -4.5453477, -4.1049075, -0.1561798, 0.1522694
5: -6.9628000, -6.3917418, -6.9572611, -6.3880696, -0.1781972, 0.1726137
6: -15.2514458, -14.7671051, -15.2481565, -14.7592335, -0.2156672, 0.2104924
7: 4.7450771, 4.9844060, 4.7422171, 4.9833989, -0.0948055, 0.0958952
8: -4.3994904, -3.8782187, -4.4112911, -3.8851871, -0.1841863, 0.1906958
9: -0.5994899, -0.2285950, -0.6040008, -0.2234738, -0.1664268, 0.1660714

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623596, upper bound: 0.0599229
time: 3.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623596, upper bound: 0.0599229
time: 2.99 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.2897396, -10.6782627, -11.2843895, -10.6633625, -0.2403682, 0.2296447
1: -3.8103223, -3.4452462, -3.8082671, -3.4397821, -0.1828558, 0.1961157
2: -8.0769577, -7.5978322, -8.0798960, -7.6132765, -0.2270699, 0.2589179
3: -1.3604417, -0.8735156, -1.3549347, -0.8743904, -0.1970690, 0.1946652
4: -4.5470915, -4.0967150, -4.5444365, -4.1049113, -0.1561357, 0.1555644
5: -6.9674592, -6.3932891, -6.9572377, -6.3886204, -0.1831189, 0.1744771
6: -15.2485161, -14.7532482, -15.2460098, -14.7592335, -0.2132984, 0.2235039
7: 4.7408648, 4.9801540, 4.7422171, 4.9818625, -0.1054348, 0.0948986
8: -4.3949509, -3.8716669, -4.4099112, -3.8853531, -0.1884968, 0.2011038
9: -0.6055264, -0.2289188, -0.6040008, -0.2238538, -0.1707808, 0.1666026

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 311

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623599, upper bound: 0.0599232
time: 3.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623599, upper bound: 0.0599232
time: 3.11 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.2857084, -10.6801167, -11.2807703, -10.6674938, -0.2393546, 0.2238880
1: -3.8090844, -3.4490786, -3.8037329, -3.4470291, -0.1895981, 0.1776742
2: -8.0822468, -7.6165447, -8.0833817, -7.6126175, -0.2286720, 0.2326424
3: -1.3594458, -0.8740344, -1.3556159, -0.8737481, -0.1953059, 0.1945575
4: -4.5504456, -4.0984235, -4.5477333, -4.1015577, -0.1532634, 0.1580436
5: -6.9630017, -6.3917418, -6.9575338, -6.3887749, -0.1847076, 0.1682255
6: -15.2513533, -14.7671051, -15.2472925, -14.7610683, -0.2185633, 0.2085764
7: 4.7450194, 4.9844060, 4.7421699, 4.9829555, -0.0976608, 0.0938260
8: -4.3994904, -3.8789439, -4.4070430, -3.8881264, -0.1776061, 0.1976521
9: -0.5994899, -0.2280638, -0.6047645, -0.2220597, -0.1650668, 0.1693525

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623627, upper bound: 0.0598262
time: 3.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623627, upper bound: 0.0598262
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.2885818, -10.6782627, -11.2807703, -10.6678295, -0.2469432, 0.2236370
1: -3.8087656, -3.4452462, -3.8024445, -3.4470291, -0.1871130, 0.1910100
2: -8.0769615, -7.5975595, -8.0802593, -7.6126175, -0.2265410, 0.2599969
3: -1.3606849, -0.8735154, -1.3556159, -0.8738604, -0.1974306, 0.1944513
4: -4.5470915, -4.0955052, -4.5468225, -4.1015615, -0.1532195, 0.1613429
5: -6.9676571, -6.3932891, -6.9575109, -6.3893290, -0.1896294, 0.1700894
6: -15.2484379, -14.7532482, -15.2451515, -14.7610683, -0.2163374, 0.2215894
7: 4.7408071, 4.9801559, 4.7421699, 4.9814205, -0.1082910, 0.0928434
8: -4.3949509, -3.8723822, -4.4056602, -3.8882923, -0.1819170, 0.2080537
9: -0.6055264, -0.2283864, -0.6047645, -0.2224383, -0.1694216, 0.1698880

Time for backsubstitution: 8.83 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 311

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623627, upper bound: 0.0598262
time: 2.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0623627, upper bound: 0.0598262
time: 3.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.2868099, -10.6808472, -11.2843895, -10.6633673, -0.2329860, 0.2296048
1: -3.8101254, -3.4488864, -3.8092799, -3.4397821, -0.1849918, 0.1821955
2: -8.0811834, -7.6163425, -8.0825062, -7.6132765, -0.2286098, 0.2303113
3: -1.3587487, -0.8736022, -1.3546281, -0.8742902, -0.1950206, 0.1955277
4: -4.5506601, -4.1007843, -4.5453477, -4.1054573, -0.1581246, 0.1519348
5: -6.9629560, -6.3944750, -6.9572592, -6.3892422, -0.1866280, 0.1719122
6: -15.2517223, -14.7651129, -15.2479839, -14.7592335, -0.2158062, 0.2096162
7: 4.7435036, 4.9845428, 4.7422171, 4.9832869, -0.0943562, 0.0962735
8: -4.4003925, -3.8784251, -4.4112911, -3.8853407, -0.1836541, 0.1907443
9: -0.6063750, -0.2153063, -0.6051831, -0.2234793, -0.1702406, 0.1791816

Time for backsubstitution: 8.76 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615904, upper bound: 0.0605756
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615904, upper bound: 0.0605756
time: 3.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.2856512, -10.6808453, -11.2807703, -10.6678352, -0.2395585, 0.2236007
1: -3.8084550, -3.4488864, -3.8034413, -3.4470291, -0.1891847, 0.1770878
2: -8.0811853, -7.6160684, -8.0828733, -7.6126175, -0.2280626, 0.2313874
3: -1.3589895, -0.8736010, -1.3553114, -0.8737602, -0.1953835, 0.1953216
4: -4.5506601, -4.0995731, -4.5477333, -4.1021061, -0.1552084, 0.1577092
5: -6.9631567, -6.3944750, -6.9575315, -6.3899498, -0.1931384, 0.1675244
6: -15.2516298, -14.7651129, -15.2471228, -14.7610683, -0.2187057, 0.2076995
7: 4.7434468, 4.9845419, 4.7421699, 4.9828444, -0.0972116, 0.0942047
8: -4.4003925, -3.8791490, -4.4070430, -3.8882771, -0.1770736, 0.1977007
9: -0.6063750, -0.2147741, -0.6059473, -0.2220640, -0.1688806, 0.1824629

Time for backsubstitution: 8.84 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 2341
type: B, layer: 3, pos: 557
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 962
type: B, layer: 3, pos: 558
type: B, layer: 3, pos: 1852
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1955
type: B, layer: 3, pos: 2568

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 312

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615935, upper bound: 0.0604870
time: 3.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0615935, upper bound: 0.0604871
time: 3.02 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.43 + 546.98 = 602.42 seconds
