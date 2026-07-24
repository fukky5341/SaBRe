## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.167531904


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.8594208, -11.0414743, -11.8594208, -11.0414743, -0.4602838, 0.4602839)
1: (2.9841757, 3.4512811, 2.9841757, 3.4512811, -0.2606032, 0.2606031)
2: (-3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2680004, 0.2680004)
3: (-11.5178366, -11.0182323, -11.5178366, -11.0182323, -0.2803777, 0.2803777)
4: (-2.7487082, -2.4019017, -2.7487082, -2.4019017, -0.1896749, 0.1896749)
5: (-9.3756046, -8.8233833, -9.3756046, -8.8233833, -0.2679567, 0.2679568)
6: (-7.3736968, -6.7107229, -7.3736968, -6.7107229, -0.4301465, 0.4301466)
7: (-4.0610304, -3.7727518, -4.0610304, -3.7727518, -0.2142508, 0.2142508)
8: (-1.4124641, -1.0182290, -1.4124641, -1.0182290, -0.2695249, 0.2695248)
9: (-14.8436832, -14.1191483, -14.8436832, -14.1191483, -0.3115454, 0.3115455)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.46 + 32.78 = 55.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1745120, upper bound: 0.1745124

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1743959, upper bound: 0.1707185
time: 2.96 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1745031, upper bound: 0.1745019
time: 2.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.15 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.15
Output dim: 1, lower bound: -0.1743959, upper bound: 0.1707185
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.15
Output dim: 1, lower bound: -0.1745031, upper bound: 0.1745019

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -11.8572254, -11.0468311, -11.8592081, -11.0431023, -0.4562225, 0.4538910
1: 2.9881830, 3.4398615, 2.9843988, 3.4476771, -0.2514273, 0.2485086
2: -3.8695643, -3.5162022, -3.8701825, -3.5157528, -0.2670341, 0.2671111
3: -11.5054388, -11.0211821, -11.5139256, -11.0182829, -0.2685342, 0.2736341
4: -2.7465434, -2.4053721, -2.7485161, -2.4029942, -0.1863275, 0.1859658
5: -9.3655853, -8.8260860, -9.3724632, -8.8233843, -0.2579345, 0.2616446
6: -7.3545876, -6.7158942, -7.3676910, -6.7107553, -0.4103003, 0.4172196
7: -4.0600548, -3.7759428, -4.0609908, -3.7737403, -0.2118479, 0.2108884
8: -1.4116793, -1.0192070, -1.4122186, -1.0184565, -0.2685202, 0.2684474
9: -14.8411798, -14.1287928, -14.8436289, -14.1221828, -0.3058017, 0.3017281

Time for backsubstitution: 20.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707191, upper bound: 0.1707190
time: 3.05 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707191, upper bound: 0.1707195
time: 3.47 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.8594189, -11.0414791, -11.8594208, -11.0414743, -0.4602835, 0.4559838
1: 2.9841771, 3.4512742, 2.9841757, 3.4512811, -0.2606018, 0.2500389
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2680004, 0.2678013
3: -11.5178308, -11.0182343, -11.5178366, -11.0182323, -0.2709417, 0.2803673
4: -2.7487078, -2.4019032, -2.7487082, -2.4019017, -0.1896747, 0.1865656
5: -9.3755960, -8.8233814, -9.3756046, -8.8233833, -0.2594401, 0.2678899
6: -7.3736877, -6.7107220, -7.3736968, -6.7107229, -0.4133227, 0.4299678
7: -4.0610294, -3.7727537, -4.0610304, -3.7727518, -0.2141813, 0.2118151
8: -1.4124637, -1.0182290, -1.4124641, -1.0182290, -0.2694663, 0.2695248
9: -14.8436852, -14.1191521, -14.8436832, -14.1191483, -0.3114693, 0.3039902

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707191, upper bound: 0.1743959
time: 2.92 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707189, upper bound: 0.1743961
time: 3.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.00 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 1, lower bound: -0.1707191, upper bound: 0.1707190
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 1, lower bound: -0.1707191, upper bound: 0.1707195
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 1, lower bound: -0.1707191, upper bound: 0.1743959
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 1, lower bound: -0.1707189, upper bound: 0.1743961

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -11.8572254, -11.0468311, -11.8572254, -11.0468311, -0.4519212, 0.4519211
1: 2.9881830, 3.4398615, 2.9881830, 3.4398615, -0.2436562, 0.2436562
2: -3.8695643, -3.5162022, -3.8695643, -3.5162022, -0.2665026, 0.2665026
3: -11.5054388, -11.0211821, -11.5054388, -11.0211821, -0.2656074, 0.2656072
4: -2.7465434, -2.4053721, -2.7465434, -2.4053721, -0.1838899, 0.1838898
5: -9.3655853, -8.8260860, -9.3655853, -8.8260860, -0.2550951, 0.2550951
6: -7.3545876, -6.7158942, -7.3545876, -6.7158942, -0.4040680, 0.4040681
7: -4.0600548, -3.7759428, -4.0600548, -3.7759428, -0.2097139, 0.2097139
8: -1.4116793, -1.0192070, -1.4116793, -1.0192070, -0.2678803, 0.2678804
9: -14.8411798, -14.1287928, -14.8411798, -14.1287928, -0.2992309, 0.2992309

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 472

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699671, upper bound: 0.1707158
time: 3.08 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707182, upper bound: 0.1707161
time: 3.10 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -11.8572254, -11.0468311, -11.8594189, -11.0414791, -0.4580922, 0.4541086
1: 2.9881830, 3.4398615, 2.9841771, 3.4512742, -0.2526174, 0.2491766
2: -3.8695643, -3.5162022, -3.8703389, -3.5155807, -0.2672405, 0.2672625
3: -11.5054388, -11.0211821, -11.5178308, -11.0182343, -0.2685795, 0.2754556
4: -2.7465434, -2.4053721, -2.7487078, -2.4019032, -0.1874448, 0.1861187
5: -9.3655853, -8.8260860, -9.3755960, -8.8233814, -0.2578988, 0.2617952
6: -7.3545876, -6.7158942, -7.3736877, -6.7107220, -0.4105711, 0.4190469
7: -4.0600548, -3.7759428, -4.0610294, -3.7727537, -0.2129500, 0.2109429
8: -1.4116793, -1.0192070, -1.4124637, -1.0182290, -0.2687008, 0.2687039
9: -14.8411798, -14.1287928, -14.8436852, -14.1191521, -0.3089937, 0.3017027

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 472

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707184, upper bound: 0.1699647
time: 3.10 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707182, upper bound: 0.1707158
time: 3.34 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.8594189, -11.0414791, -11.8572254, -11.0468311, -0.4541087, 0.4580923
1: 2.9841771, 3.4512742, 2.9881830, 3.4398615, -0.2491765, 0.2526174
2: -3.8703389, -3.5155807, -3.8695643, -3.5162022, -0.2672622, 0.2672408
3: -11.5178308, -11.0182343, -11.5054388, -11.0211821, -0.2754557, 0.2685794
4: -2.7487078, -2.4019032, -2.7465434, -2.4053721, -0.1861188, 0.1874449
5: -9.3755960, -8.8233814, -9.3655853, -8.8260860, -0.2617952, 0.2578988
6: -7.3736877, -6.7107220, -7.3545876, -6.7158942, -0.4190469, 0.4105712
7: -4.0610294, -3.7727537, -4.0600548, -3.7759428, -0.2109429, 0.2129500
8: -1.4124637, -1.0182290, -1.4116793, -1.0192070, -0.2687041, 0.2687008
9: -14.8436852, -14.1191521, -14.8411798, -14.1287928, -0.3017026, 0.3089937

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 472

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699648, upper bound: 0.1743917
time: 3.07 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707158, upper bound: 0.1743920
time: 3.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.8594189, -11.0414791, -11.8594189, -11.0414791, -0.4559834, 0.4559834
1: 2.9841771, 3.4512742, 2.9841771, 3.4512742, -0.2500379, 0.2500379
2: -3.8703389, -3.5155807, -3.8703389, -3.5155807, -0.2678010, 0.2678010
3: -11.5178308, -11.0182343, -11.5178308, -11.0182343, -0.2709415, 0.2709414
4: -2.7487078, -2.4019032, -2.7487078, -2.4019032, -0.1865655, 0.1865655
5: -9.3755960, -8.8233814, -9.3755960, -8.8233814, -0.2594398, 0.2594398
6: -7.3736877, -6.7107220, -7.3736877, -6.7107220, -0.4133213, 0.4133213
7: -4.0610294, -3.7727537, -4.0610294, -3.7727537, -0.2118148, 0.2118149
8: -1.4124637, -1.0182290, -1.4124637, -1.0182290, -0.2694659, 0.2694662
9: -14.8436852, -14.1191521, -14.8436852, -14.1191521, -0.3039900, 0.3039899

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 472

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707159, upper bound: 0.1737486
time: 4.13 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707159, upper bound: 0.1743927
time: 3.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.69 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1699671, upper bound: 0.1707158
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1707182, upper bound: 0.1707161
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1707184, upper bound: 0.1699647
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1707182, upper bound: 0.1707158
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1699648, upper bound: 0.1743917
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1707158, upper bound: 0.1743920
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1707159, upper bound: 0.1737486
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 28.69
Output dim: 1, lower bound: -0.1707159, upper bound: 0.1743927

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.8571177, -11.0470295, -11.8572254, -11.0468311, -0.4511089, 0.4508705
1: 2.9889760, 3.4398234, 2.9881830, 3.4398615, -0.2428381, 0.2436225
2: -3.8678546, -3.5162549, -3.8695643, -3.5162022, -0.2648013, 0.2664483
3: -11.5053701, -11.0230007, -11.5054388, -11.0211821, -0.2655368, 0.2637835
4: -2.7451499, -2.4054050, -2.7465434, -2.4053721, -0.1824349, 0.1838533
5: -9.3655128, -8.8275595, -9.3655853, -8.8260860, -0.2548752, 0.2534652
6: -7.3543868, -6.7158942, -7.3545876, -6.7158942, -0.4035361, 0.4037799
7: -4.0596118, -3.7759550, -4.0600548, -3.7759428, -0.2091755, 0.2096308
8: -1.4116664, -1.0209103, -1.4116793, -1.0192070, -0.2678655, 0.2661531
9: -14.8410168, -14.1288404, -14.8411798, -14.1287928, -0.2988427, 0.2990085

Time for backsubstitution: 20.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 472

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699671, upper bound: 0.1699664
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699671, upper bound: 0.1707178
time: 3.06 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.8599606, -11.0462456, -11.8572273, -11.0468321, -0.4529564, 0.4563632
1: 2.9879899, 3.4434085, 2.9881840, 3.4398623, -0.2439821, 0.2472045
2: -3.8699586, -3.5089617, -3.8695612, -3.5162032, -0.2674317, 0.2719498
3: -11.5135765, -11.0211334, -11.5054398, -11.0211830, -0.2713196, 0.2661117
4: -2.7472782, -2.3982887, -2.7465420, -2.4053721, -0.1857885, 0.1910245
5: -9.3726282, -8.8260918, -9.3655863, -8.8260870, -0.2586942, 0.2561609
6: -7.3563995, -6.7150726, -7.3545895, -6.7158942, -0.4044535, 0.4061372
7: -4.0602055, -3.7737751, -4.0600529, -3.7759426, -0.2104521, 0.2117339
8: -1.4189196, -1.0183563, -1.4116783, -1.0192099, -0.2734768, 0.2682858
9: -14.8411818, -14.1267223, -14.8411798, -14.1287928, -0.3002362, 0.3006109

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 472

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707184, upper bound: 0.1699668
time: 3.13 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707184, upper bound: 0.1707183
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -11.8572254, -11.0468311, -11.8593102, -11.0416794, -0.4570410, 0.4532963
1: 2.9881830, 3.4398615, 2.9849691, 3.4512367, -0.2525858, 0.2483588
2: -3.8695643, -3.5162022, -3.8686304, -3.5156333, -0.2671864, 0.2655621
3: -11.5054388, -11.0211821, -11.5177612, -11.0200520, -0.2667553, 0.2753866
4: -2.7465434, -2.4053721, -2.7473152, -2.4019353, -0.1874085, 0.1846648
5: -9.3655853, -8.8260860, -9.3755255, -8.8248568, -0.2562689, 0.2615763
6: -7.3545876, -6.7158942, -7.3734889, -6.7107220, -0.4102831, 0.4185041
7: -4.0600548, -3.7759428, -4.0605860, -3.7727668, -0.2128671, 0.2104039
8: -1.4116793, -1.0192070, -1.4124527, -1.0199318, -0.2669735, 0.2686884
9: -14.8411798, -14.1287928, -14.8435230, -14.1192017, -0.3087711, 0.3013144

Time for backsubstitution: 20.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 472

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1699643
time: 3.27 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1699642
time: 3.26 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -11.8572273, -11.0468321, -11.8621559, -11.0409069, -0.4625187, 0.4551576
1: 2.9881840, 3.4398623, 2.9839854, 3.4548206, -0.2531216, 0.2495013
2: -3.8695612, -3.5162032, -3.8707328, -3.5083382, -0.2723682, 0.2681906
3: -11.5054398, -11.0211830, -11.5259714, -11.0181875, -0.2690843, 0.2763009
4: -2.7465420, -2.4053721, -2.7494607, -2.3948197, -0.1933312, 0.1880257
5: -9.3655863, -8.8260870, -9.3826380, -8.8233871, -0.2584881, 0.2623485
6: -7.3545895, -6.7158942, -7.3754935, -6.7099004, -0.4126406, 0.4193337
7: -4.0600529, -3.7759426, -4.0611806, -3.7705896, -0.2149686, 0.2116807
8: -1.4116783, -1.0192099, -1.4197059, -1.0173879, -0.2691083, 0.2738159
9: -14.8411798, -14.1287928, -14.8436823, -14.1170826, -0.3103781, 0.3027081

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 472

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1707155
time: 3.15 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1707162
time: 3.19 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.8593102, -11.0416794, -11.8572254, -11.0468311, -0.4532964, 0.4570409
1: 2.9849691, 3.4512367, 2.9881830, 3.4398615, -0.2483587, 0.2525858
2: -3.8686304, -3.5156333, -3.8695643, -3.5162022, -0.2655618, 0.2671862
3: -11.5177612, -11.0200520, -11.5054388, -11.0211821, -0.2753865, 0.2667553
4: -2.7473152, -2.4019353, -2.7465434, -2.4053721, -0.1846647, 0.1874085
5: -9.3755255, -8.8248568, -9.3655853, -8.8260860, -0.2615763, 0.2562689
6: -7.3734889, -6.7107220, -7.3545876, -6.7158942, -0.4185042, 0.4102832
7: -4.0605860, -3.7727668, -4.0600548, -3.7759428, -0.2104039, 0.2128671
8: -1.4124527, -1.0199318, -1.4116793, -1.0192070, -0.2686886, 0.2669735
9: -14.8435230, -14.1192017, -14.8411798, -14.1287928, -0.3013142, 0.3087713

Time for backsubstitution: 20.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 472

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699646, upper bound: 0.1736402
time: 2.95 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699646, upper bound: 0.1743916
time: 2.99 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.8621559, -11.0409069, -11.8572273, -11.0468321, -0.4551578, 0.4625187
1: 2.9839854, 3.4548206, 2.9881840, 3.4398623, -0.2495013, 0.2531216
2: -3.8707328, -3.5083382, -3.8695612, -3.5162032, -0.2681906, 0.2723682
3: -11.5259714, -11.0181875, -11.5054398, -11.0211830, -0.2763008, 0.2690842
4: -2.7494607, -2.3948197, -2.7465420, -2.4053721, -0.1880258, 0.1933312
5: -9.3826380, -8.8233871, -9.3655863, -8.8260870, -0.2623487, 0.2584881
6: -7.3754935, -6.7099004, -7.3545895, -6.7158942, -0.4193337, 0.4126406
7: -4.0611806, -3.7705896, -4.0600529, -3.7759426, -0.2116805, 0.2149686
8: -1.4197059, -1.0173879, -1.4116783, -1.0192099, -0.2738159, 0.2691083
9: -14.8436823, -14.1170826, -14.8411798, -14.1287928, -0.3027081, 0.3103781

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 472

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707161, upper bound: 0.1736404
time: 3.10 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707161, upper bound: 0.1743918
time: 3.07 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -11.8594189, -11.0414791, -11.8593102, -11.0416794, -0.4549315, 0.4551709
1: 2.9841771, 3.4512742, 2.9849691, 3.4512367, -0.2500045, 0.2492200
2: -3.8703389, -3.5155807, -3.8686304, -3.5156333, -0.2677469, 0.2661004
3: -11.5178308, -11.0182343, -11.5177612, -11.0200520, -0.2691175, 0.2708710
4: -2.7487078, -2.4019032, -2.7473152, -2.4019353, -0.1865290, 0.1851117
5: -9.3755960, -8.8233814, -9.3755255, -8.8248568, -0.2578099, 0.2592208
6: -7.3736877, -6.7107220, -7.3734889, -6.7107220, -0.4130332, 0.4127898
7: -4.0610294, -3.7727537, -4.0605860, -3.7727668, -0.2117321, 0.2112757
8: -1.4124637, -1.0182290, -1.4124527, -1.0199318, -0.2677389, 0.2694509
9: -14.8436852, -14.1191521, -14.8435230, -14.1192017, -0.3037679, 0.3036015

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 472
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 472

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1737477
time: 3.13 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1737492
time: 3.00 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -11.8594208, -11.0414772, -11.8621559, -11.0409069, -0.4604089, 0.4570328
1: 2.9841771, 3.4512734, 2.9839854, 3.4548206, -0.2535886, 0.2503625
2: -3.8703363, -3.5155811, -3.8707328, -3.5083382, -0.2732885, 0.2687294
3: -11.5178308, -11.0182362, -11.5259714, -11.0181875, -0.2714467, 0.2766905
4: -2.7487059, -2.4019027, -2.7494607, -2.3948197, -0.1937010, 0.1884726
5: -9.3755970, -8.8233843, -9.3826380, -8.8233871, -0.2605062, 0.2631681
6: -7.3736877, -6.7107220, -7.3754935, -6.7099004, -0.4153898, 0.4137076
7: -4.0610280, -3.7727537, -4.0611806, -3.7705896, -0.2138375, 0.2125522
8: -1.4124632, -1.0182304, -1.4197059, -1.0173879, -0.2698735, 0.2749504
9: -14.8436842, -14.1191521, -14.8436823, -14.1170826, -0.3053784, 0.3049951

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 472

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1744996
time: 3.14 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1745002
time: 3.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.33 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1699671, upper bound: 0.1699664
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1699671, upper bound: 0.1707178
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1707184, upper bound: 0.1699668
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1707184, upper bound: 0.1707183
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1699643
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1699642
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1707155
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1736408, upper bound: 0.1707162
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1699646, upper bound: 0.1736402
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1699646, upper bound: 0.1743916
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1707161, upper bound: 0.1736404
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1707161, upper bound: 0.1743918
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1737477
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1737492
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1744996
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.33
Output dim: 1, lower bound: -0.1701205, upper bound: 0.1745002

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.8571177, -11.0470295, -11.8571177, -11.0470295, -0.4500585, 0.4500583
1: 2.9889760, 3.4398234, 2.9889760, 3.4398234, -0.2428045, 0.2428045
2: -3.8678546, -3.5162549, -3.8678546, -3.5162549, -0.2647467, 0.2647467
3: -11.5053701, -11.0230007, -11.5053701, -11.0230007, -0.2637128, 0.2637128
4: -2.7451499, -2.4054050, -2.7451499, -2.4054050, -0.1823984, 0.1823984
5: -9.3655128, -8.8275595, -9.3655128, -8.8275595, -0.2532454, 0.2532453
6: -7.3543868, -6.7158942, -7.3543868, -6.7158942, -0.4032481, 0.4032482
7: -4.0596118, -3.7759550, -4.0596118, -3.7759550, -0.2090924, 0.2090924
8: -1.4116664, -1.0209103, -1.4116664, -1.0209103, -0.2661383, 0.2661383
9: -14.8410168, -14.1288404, -14.8410168, -14.1288404, -0.2986203, 0.2986203

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698225, upper bound: 0.1683191
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699664, upper bound: 0.1699661
time: 3.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.8571177, -11.0470295, -11.8599606, -11.0462456, -0.4509163, 0.4519075
1: 2.9889760, 3.4398234, 2.9879899, 3.4434085, -0.2463875, 0.2437708
2: -3.8678546, -3.5162549, -3.8699586, -3.5089617, -0.2702460, 0.2670062
3: -11.5053701, -11.0230007, -11.5135765, -11.0211334, -0.2655569, 0.2694931
4: -2.7451499, -2.4054050, -2.7472782, -2.3982887, -0.1895718, 0.1848750
5: -9.3655128, -8.8275595, -9.3726282, -8.8260918, -0.2547177, 0.2570630
6: -7.3543868, -6.7158942, -7.3563995, -6.7150726, -0.4040692, 0.4041657
7: -4.0596118, -3.7759550, -4.0602055, -3.7737751, -0.2111959, 0.2097005
8: -1.4116664, -1.0209103, -1.4189196, -1.0183563, -0.2679863, 0.2717444
9: -14.8410168, -14.1288404, -14.8411818, -14.1267223, -0.3002231, 0.2987907

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1698225, upper bound: 0.1690704
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1699664, upper bound: 0.1707175
time: 3.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.8599606, -11.0462456, -11.8571177, -11.0470295, -0.4519076, 0.4509161
1: 2.9879899, 3.4434085, 2.9889760, 3.4398234, -0.2437708, 0.2463876
2: -3.8699586, -3.5089617, -3.8678546, -3.5162549, -0.2670064, 0.2702460
3: -11.5135765, -11.0211334, -11.5053701, -11.0230007, -0.2694931, 0.2655568
4: -2.7472782, -2.3982887, -2.7451499, -2.4054050, -0.1848751, 0.1895718
5: -9.3726282, -8.8260918, -9.3655128, -8.8275595, -0.2570630, 0.2547178
6: -7.3563995, -6.7150726, -7.3543868, -6.7158942, -0.4041657, 0.4040691
7: -4.0602055, -3.7737751, -4.0596118, -3.7759550, -0.2097005, 0.2111959
8: -1.4189196, -1.0183563, -1.4116664, -1.0209103, -0.2717443, 0.2679861
9: -14.8411818, -14.1267223, -14.8410168, -14.1288404, -0.2987908, 0.3002231

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1690701, upper bound: 0.1698222
time: 3.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707177, upper bound: 0.1699664
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.8599606, -11.0462456, -11.8599606, -11.0462456, -0.4566207, 0.4566208
1: 2.9879899, 3.4434085, 2.9879899, 3.4434085, -0.2448732, 0.2448732
2: -3.8699586, -3.5089617, -3.8699586, -3.5089617, -0.2691064, 0.2691064
3: -11.5135765, -11.0211334, -11.5135765, -11.0211334, -0.2672633, 0.2672633
4: -2.7472782, -2.3982887, -2.7472782, -2.3982887, -0.1871216, 0.1871215
5: -9.3726282, -8.8260918, -9.3726282, -8.8260918, -0.2568156, 0.2568156
6: -7.3563995, -6.7150726, -7.3563995, -6.7150726, -0.4068096, 0.4068097
7: -4.0602055, -3.7737751, -4.0602055, -3.7737751, -0.2106278, 0.2106278
8: -1.4189196, -1.0183563, -1.4189196, -1.0183563, -0.2712224, 0.2712226
9: -14.8411818, -14.1267223, -14.8411818, -14.1267223, -0.3007209, 0.3007208

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1690704, upper bound: 0.1698222
time: 3.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1707179, upper bound: 0.1699662
time: 3.21 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -11.8571177, -11.0470295, -11.8593102, -11.0416794, -0.4562287, 0.4522458
1: 2.9889760, 3.4398234, 2.9849691, 3.4512367, -0.2517688, 0.2483252
2: -3.8678546, -3.5162549, -3.8686304, -3.5156333, -0.2654848, 0.2655072
3: -11.5053701, -11.0230007, -11.5177612, -11.0200520, -0.2666848, 0.2735602
4: -2.7451499, -2.4054050, -2.7473152, -2.4019353, -0.1859537, 0.1846284
5: -9.3655128, -8.8275595, -9.3755255, -8.8248568, -0.2560489, 0.2599449
6: -7.3543868, -6.7158942, -7.3734889, -6.7107220, -0.4097514, 0.4182161
7: -4.0596118, -3.7759550, -4.0605860, -3.7727668, -0.2123287, 0.2103208
8: -1.4116664, -1.0209103, -1.4124527, -1.0199318, -0.2669585, 0.2669613
9: -14.8410168, -14.1288404, -14.8435230, -14.1192017, -0.3083830, 0.3010919

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1719932, upper bound: 0.1698196
time: 3.12 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1736403, upper bound: 0.1699637
time: 3.19 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -11.8599606, -11.0462456, -11.8593102, -11.0416794, -0.4580777, 0.4531035
1: 2.9879899, 3.4434085, 2.9849691, 3.4512367, -0.2527379, 0.2507744
2: -3.8699586, -3.5089617, -3.8686304, -3.5156333, -0.2677443, 0.2708459
3: -11.5135765, -11.0211334, -11.5177612, -11.0200520, -0.2696763, 0.2754146
4: -2.7472782, -2.3982887, -2.7473152, -2.4019353, -0.1884302, 0.1915171
5: -9.3726282, -8.8260918, -9.3755255, -8.8248568, -0.2571778, 0.2614226
6: -7.3563995, -6.7150726, -7.3734889, -6.7107220, -0.4106688, 0.4182159
7: -4.0602055, -3.7737751, -4.0605860, -3.7727668, -0.2129366, 0.2124244
8: -1.4189196, -1.0183563, -1.4124527, -1.0199318, -0.2723930, 0.2688093
9: -14.8411818, -14.1267223, -14.8435230, -14.1192017, -0.3085535, 0.3026947

Time for backsubstitution: 22.00 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.24 + 551.81 = 607.05 seconds
