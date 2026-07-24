## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13366404


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2675802, 0.2675802)
1: (2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2295630, 0.2295631)
2: (-4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2331604, 0.2331605)
3: (-14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4674888, 0.4674888)
4: (-3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2475597, 0.2475598)
5: (-8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3414524, 0.3414524)
6: (-4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2934346, 0.2934346)
7: (-8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3316460, 0.3316460)
8: (-1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006594, 0.3006594)
9: (-7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2711308, 0.2711308)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.46 + 32.45 = 54.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1452866, upper bound: 0.1452870

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1452856
time: 2.83 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1452855
time: 2.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.82 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.82
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1452856
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.82
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1452855

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.6445098, -6.9111762, -7.6448908, -6.9110246, -0.2671475, 0.2672306
1: 2.5627270, 3.0936780, 2.5623522, 3.0937791, -0.2290163, 0.2293018
2: -4.9284382, -4.3580036, -4.9287162, -4.3579311, -0.2328573, 0.2330683
3: -14.4969034, -13.4907455, -14.4969625, -13.4901037, -0.4673562, 0.4667497
4: -3.0553031, -2.4094348, -3.0554028, -2.4088497, -0.2474661, 0.2469420
5: -8.6011906, -7.7889872, -8.6014805, -7.7889137, -0.3410733, 0.3412795
6: -4.5142226, -3.8795028, -4.5145478, -3.8793623, -0.2930806, 0.2931465
7: -8.2882624, -7.7436323, -8.2884521, -7.7425776, -0.3311586, 0.3304058
8: -1.2434819, -0.5141721, -1.2437325, -0.5141001, -0.3003552, 0.3006004
9: -7.3486681, -6.6414757, -7.3487859, -6.6409111, -0.2709333, 0.2704828

Time for backsubstitution: 21.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1419451, upper bound: 0.1417045
time: 2.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417803
time: 2.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.6452475, -6.9100342, -7.6448898, -6.9110250, -0.2677459, 0.2679796
1: 2.5614853, 3.0961642, 2.5623550, 3.0937786, -0.2316538, 0.2315585
2: -4.9295120, -4.3562870, -4.9287138, -4.3579321, -0.2344437, 0.2347729
3: -14.5021038, -13.4898701, -14.4969597, -13.4901056, -0.4723690, 0.4683657
4: -3.0601912, -2.4085894, -3.0554018, -2.4088516, -0.2517862, 0.2477689
5: -8.6015844, -7.7864594, -8.6014748, -7.7889156, -0.3422043, 0.3436801
6: -4.5146618, -3.8774104, -4.5145454, -3.8793643, -0.2935550, 0.2951832
7: -8.2967701, -7.7418885, -8.2884502, -7.7425795, -0.3351612, 0.3331391
8: -1.2443509, -0.5129433, -1.2437305, -0.5141015, -0.3009868, 0.3022424
9: -7.3521113, -6.6404891, -7.3487830, -6.6409168, -0.2735946, 0.2713040

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1421322
time: 3.22 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417802
time: 3.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.98 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.98
Output dim: 1, lower bound: -0.1419451, upper bound: 0.1417045
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.98
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417803
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 28.98
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1421322
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 28.98
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417802

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.6441450, -6.9191666, -7.6441545, -6.9274426, -0.2403988, 0.2529381
1: 2.5648222, 3.0886507, 2.5663915, 3.0838852, -0.2114333, 0.2126209
2: -4.9278183, -4.3598652, -4.9274607, -4.3614988, -0.2276909, 0.2286100
3: -14.4962807, -13.4920635, -14.4957085, -13.4927607, -0.4640741, 0.4644079
4: -3.0510044, -2.4098778, -3.0466671, -2.4097390, -0.2416396, 0.2387161
5: -8.5997419, -7.7895613, -8.5985060, -7.7900581, -0.3387990, 0.3381546
6: -4.5134807, -3.8855560, -4.5131145, -3.8917797, -0.2797346, 0.2846801
7: -8.2854271, -7.7444377, -8.2827616, -7.7442083, -0.3214748, 0.3208021
8: -1.2381215, -0.5150447, -1.2326212, -0.5158162, -0.2907951, 0.2855395
9: -7.3464046, -6.6434131, -7.3443680, -6.6449294, -0.2605752, 0.2585249

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
time: 3.11 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417048
time: 3.08 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.6442089, -6.9244566, -7.6523972, -6.9356070, -0.2419162, 0.2878424
1: 2.5649738, 3.0861835, 2.5644341, 3.0803323, -0.2145492, 0.2267345
2: -4.9275732, -4.3601608, -4.9274716, -4.3614545, -0.2278490, 0.2281951
3: -14.4963703, -13.4926023, -14.4977264, -13.4934292, -0.4639308, 0.4655392
4: -3.0505633, -2.4099212, -3.0469060, -2.4025278, -0.2460800, 0.2396663
5: -8.5995684, -7.7894506, -8.5984735, -7.7876372, -0.3393245, 0.3382246
6: -4.5135460, -3.8890085, -4.5210166, -3.8959651, -0.2819953, 0.2876258
7: -8.2869234, -7.7447462, -8.2869110, -7.7421656, -0.3195438, 0.3288517
8: -1.2343161, -0.5150528, -1.2267947, -0.5115714, -0.3039551, 0.2853699
9: -7.3463516, -6.6438594, -7.3465843, -6.6452618, -0.2612170, 0.2622604

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 67

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417736
time: 3.34 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417398, upper bound: 0.1417799
time: 3.39 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -7.6445446, -6.9258571, -7.6445298, -6.9189038, -0.2535729, 0.2423806
1: 2.5656366, 3.0866508, 2.5644341, 3.0888195, -0.2150658, 0.2147030
2: -4.9283214, -4.3598437, -4.9281020, -4.3597784, -0.2300239, 0.2296526
3: -14.5008678, -13.4926138, -14.4963503, -13.4914160, -0.4700615, 0.4651489
4: -3.0517821, -2.4094529, -3.0511637, -2.4092903, -0.2431992, 0.2419949
5: -8.5985861, -7.7876010, -8.6000385, -7.7894831, -0.3391798, 0.3414099
6: -4.5131898, -3.8893316, -4.5138087, -3.8853357, -0.2850271, 0.2825273
7: -8.2912483, -7.7434464, -8.2856503, -7.7433767, -0.3255146, 0.3236682
8: -1.2328823, -0.5146537, -1.2383971, -0.5149646, -0.2857907, 0.2927196
9: -7.3476920, -6.6446052, -7.3465338, -6.6428423, -0.2615376, 0.2609638

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417459
time: 3.03 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417733
time: 3.08 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -7.6540842, -6.9353251, -7.6445851, -6.9242525, -0.2906675, 0.2427778
1: 2.5637808, 3.0812821, 2.5645866, 3.0862288, -0.2302401, 0.2157797
2: -4.9281354, -4.3600812, -4.9278450, -4.3600912, -0.2294133, 0.2294638
3: -14.5029688, -13.4932823, -14.4964266, -13.4919538, -0.4712653, 0.4647565
4: -3.0505385, -2.4022851, -3.0506167, -2.4093399, -0.2438579, 0.2472353
5: -8.5984716, -7.7849202, -8.5998535, -7.7893772, -0.3390791, 0.3421805
6: -4.5211654, -3.8957918, -4.5138755, -3.8889279, -0.2901671, 0.2824335
7: -8.2947931, -7.7415829, -8.2870903, -7.7436986, -0.3332214, 0.3218015
8: -1.2269917, -0.5086613, -1.2345893, -0.5149813, -0.2855690, 0.3074762
9: -7.3502388, -6.6449623, -7.3464656, -6.6432924, -0.2652012, 0.2615591

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 67

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417459
time: 2.95 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417804
time: 2.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.00 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417048
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417736
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417398, upper bound: 0.1417799
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417459
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417733
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417459
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.00
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417804

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6441545, -6.9274426, -0.2400299, 0.2404305
1: 2.5667019, 3.0838671, 2.5663915, 3.0838852, -0.2071801, 0.2075635
2: -4.9271917, -4.3615236, -4.9274607, -4.3614988, -0.2265370, 0.2267936
3: -14.4956598, -13.4933548, -14.4957085, -13.4927607, -0.4636214, 0.4630342
4: -3.0466380, -2.4103155, -3.0466671, -2.4097390, -0.2383970, 0.2378232
5: -8.5982609, -7.7901149, -8.5985060, -7.7900581, -0.3374929, 0.3376930
6: -4.5128145, -3.8918164, -4.5131145, -3.8917797, -0.2787409, 0.2790066
7: -8.2826242, -7.7452507, -8.2827616, -7.7442083, -0.3184037, 0.3176018
8: -1.2325611, -0.5158648, -1.2326212, -0.5158162, -0.2844763, 0.2845142
9: -7.3443165, -6.6454306, -7.3443680, -6.6449294, -0.2563474, 0.2558656

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1419454, upper bound: 0.1417043
time: 3.02 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417042
time: 3.05 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6441545, -6.9274426, -0.2638153, 0.2635077
1: 2.5646744, 3.0803094, 2.5663915, 3.0838852, -0.2166420, 0.2146392
2: -4.9272232, -4.3615274, -4.9274607, -4.3614988, -0.2261205, 0.2272820
3: -14.4976921, -13.4940872, -14.4957085, -13.4927607, -0.4651687, 0.4626832
4: -3.0468817, -2.4029546, -3.0466671, -2.4097390, -0.2379949, 0.2427802
5: -8.5981750, -7.7877026, -8.5985060, -7.7900581, -0.3371110, 0.3385847
6: -4.5209613, -3.8960366, -4.5131145, -3.8917797, -0.2844882, 0.2749181
7: -8.2867680, -7.7431898, -8.2827616, -7.7442083, -0.3187802, 0.3169621
8: -1.2264655, -0.5116353, -1.2326212, -0.5158162, -0.2925262, 0.2940899
9: -7.3464637, -6.6458454, -7.3443680, -6.6449294, -0.2584845, 0.2586713

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1419451, upper bound: 0.1417045
time: 2.92 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417041
time: 2.99 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6523972, -6.9356070, -0.2634575, 0.2640160
1: 2.5667019, 3.0838671, 2.5644341, 3.0803323, -0.2144547, 0.2170975
2: -4.9271917, -4.3615236, -4.9274716, -4.3614545, -0.2271098, 0.2263775
3: -14.4956598, -13.4933548, -14.4977264, -13.4934292, -0.4632947, 0.4645429
4: -3.0466380, -2.4103155, -3.0469060, -2.4025278, -0.2433007, 0.2375077
5: -8.5982609, -7.7901149, -8.5984735, -7.7876372, -0.3383553, 0.3373137
6: -4.5128145, -3.8918164, -4.5210166, -3.8959651, -0.2749739, 0.2847540
7: -8.2826242, -7.7452507, -8.2869110, -7.7421656, -0.3177633, 0.3180513
8: -1.2325611, -0.5158648, -1.2267947, -0.5115714, -0.2940755, 0.2928376
9: -7.3443165, -6.6454306, -7.3465843, -6.6452618, -0.2592442, 0.2581133

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417042
time: 3.07 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417735
time: 3.15 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6523972, -6.9356070, -0.2419795, 0.2419742
1: 2.5646744, 3.0803094, 2.5644341, 3.0803323, -0.2123000, 0.2125844
2: -4.9272232, -4.3615274, -4.9274716, -4.3614545, -0.2267520, 0.2269254
3: -14.4976921, -13.4940872, -14.4977264, -13.4934292, -0.4649186, 0.4642749
4: -3.0468817, -2.4029546, -3.0469060, -2.4025278, -0.2394452, 0.2390200
5: -8.5981750, -7.7877026, -8.5984735, -7.7876372, -0.3385239, 0.3387561
6: -4.5209613, -3.8960366, -4.5210166, -3.8959651, -0.2815423, 0.2815400
7: -8.2867680, -7.7431898, -8.2869110, -7.7421656, -0.3282247, 0.3274683
8: -1.2264655, -0.5116353, -1.2267947, -0.5115714, -0.2846923, 0.2849787
9: -7.3464637, -6.6458454, -7.3465843, -6.6452618, -0.2583287, 0.2579058

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417041
time: 3.00 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417803
time: 2.84 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -7.6445446, -6.9258571, -7.6441507, -6.9274454, -0.2406595, 0.2420011
1: 2.5656366, 3.0866508, 2.5663924, 3.0838861, -0.2097607, 0.2102909
2: -4.9283214, -4.3598437, -4.9274578, -4.3614979, -0.2281343, 0.2284646
3: -14.5008678, -13.4926138, -14.4957075, -13.4927616, -0.4686420, 0.4646804
4: -3.0517821, -2.4094529, -3.0466671, -2.4097414, -0.2422761, 0.2386502
5: -8.5985861, -7.7876010, -8.5985041, -7.7900586, -0.3387001, 0.3400664
6: -4.5131898, -3.8893316, -4.5131135, -3.8917794, -0.2791232, 0.2814918
7: -8.2912483, -7.7434464, -8.2827606, -7.7442112, -0.3222384, 0.3204740
8: -1.2328823, -0.5146537, -1.2326210, -0.5158167, -0.2847269, 0.2861435
9: -7.3476920, -6.6446052, -7.3443675, -6.6449332, -0.2587545, 0.2565745

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1421322
time: 2.97 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417459
time: 2.98 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -7.6445446, -6.9258571, -7.6523952, -6.9356079, -0.2640829, 0.2655874
1: 2.5656366, 3.0866508, 2.5644369, 3.0803323, -0.2170335, 0.2198264
2: -4.9283214, -4.3598437, -4.9274712, -4.3614559, -0.2287065, 0.2280482
3: -14.5008678, -13.4926138, -14.4977226, -13.4934320, -0.4683142, 0.4661889
4: -3.0517821, -2.4094529, -3.0469065, -2.4025297, -0.2475059, 0.2383339
5: -8.5985861, -7.7876010, -8.5984707, -7.7876391, -0.3395618, 0.3396859
6: -4.5131898, -3.8893316, -4.5210142, -3.8959672, -0.2753527, 0.2872396
7: -8.2912483, -7.7434464, -8.2869110, -7.7421694, -0.3216343, 0.3209227
8: -1.2328823, -0.5146537, -1.2267928, -0.5115724, -0.2943254, 0.2944634
9: -7.3476920, -6.6446052, -7.3465834, -6.6452646, -0.2616356, 0.2588211

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1421322
time: 3.22 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417733
time: 3.46 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -7.6540842, -6.9353251, -7.6441507, -6.9274454, -0.2664983, 0.2640729
1: 2.5637808, 3.0812821, 2.5663924, 3.0838861, -0.2203890, 0.2155925
2: -4.9281354, -4.3600812, -4.9274578, -4.3614979, -0.2275231, 0.2287176
3: -14.5029688, -13.4932823, -14.4957075, -13.4927616, -0.4702473, 0.4639654
4: -3.0505385, -2.4022851, -3.0466671, -2.4097414, -0.2417727, 0.2443849
5: -8.5984716, -7.7849202, -8.5985041, -7.7900586, -0.3378861, 0.3412042
6: -4.5211654, -3.8957918, -4.5131135, -3.8917794, -0.2870246, 0.2751503
7: -8.2947931, -7.7415829, -8.2827606, -7.7442112, -0.3226385, 0.3199033
8: -1.2269917, -0.5086613, -1.2326210, -0.5158167, -0.2929780, 0.2972099
9: -7.3502388, -6.6449623, -7.3443675, -6.6449332, -0.2609873, 0.2594118

Time for backsubstitution: 23.03 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417459
time: 3.32 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417459
time: 3.33 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -7.6540842, -6.9353251, -7.6523952, -6.9356079, -0.2436460, 0.2428158
1: 2.5637808, 3.0812821, 2.5644369, 3.0803323, -0.2140811, 0.2135190
2: -4.9281354, -4.3600812, -4.9274712, -4.3614559, -0.2279423, 0.2283587
3: -14.5029688, -13.4932823, -14.4977226, -13.4934320, -0.4699974, 0.4657133
4: -3.0505385, -2.4022851, -3.0469065, -2.4025297, -0.2432152, 0.2396014
5: -8.5984716, -7.7849202, -8.5984707, -7.7876391, -0.3395994, 0.3413749
6: -4.5211654, -3.8957918, -4.5210142, -3.8959672, -0.2816582, 0.2819843
7: -8.2947931, -7.7415829, -8.2869110, -7.7421694, -0.3318336, 0.3294262
8: -1.2269917, -0.5086613, -1.2267928, -0.5115724, -0.2851717, 0.2880182
9: -7.3502388, -6.6449623, -7.3465834, -6.6452646, -0.2605063, 0.2586822

Time for backsubstitution: 23.06 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417733
time: 3.41 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417802
time: 3.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.01 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1419454, upper bound: 0.1417043
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417042
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1419451, upper bound: 0.1417045
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417041
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417042
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417735
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417041
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417803
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1421322
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417459
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1421322
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417733
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417459
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417459
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417459, upper bound: 0.1417733
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.01
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417802

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6441545, -6.9274426, -0.2400299, 0.2404305
1: 2.5667019, 3.0838671, 2.5663915, 3.0838852, -0.2071801, 0.2075635
2: -4.9271917, -4.3615236, -4.9274607, -4.3614988, -0.2265370, 0.2267936
3: -14.4956598, -13.4933548, -14.4957085, -13.4927607, -0.4636214, 0.4630342
4: -3.0466380, -2.4103155, -3.0466671, -2.4097390, -0.2383970, 0.2378232
5: -8.5982609, -7.7901149, -8.5985060, -7.7900581, -0.3374929, 0.3376930
6: -4.5128145, -3.8918164, -4.5131145, -3.8917797, -0.2787409, 0.2790066
7: -8.2826242, -7.7452507, -8.2827616, -7.7442083, -0.3184037, 0.3176018
8: -1.2325611, -0.5158648, -1.2326212, -0.5158162, -0.2844763, 0.2845142
9: -7.3443165, -6.6454306, -7.3443680, -6.6449294, -0.2563474, 0.2558656

Time for backsubstitution: 23.11 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1421323
time: 3.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
time: 3.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6523972, -6.9356070, -0.2634575, 0.2640160
1: 2.5667019, 3.0838671, 2.5644341, 3.0803323, -0.2144547, 0.2170975
2: -4.9271917, -4.3615236, -4.9274716, -4.3614545, -0.2271098, 0.2263775
3: -14.4956598, -13.4933548, -14.4977264, -13.4934292, -0.4632947, 0.4645429
4: -3.0466380, -2.4103155, -3.0469060, -2.4025278, -0.2433007, 0.2375077
5: -8.5982609, -7.7901149, -8.5984735, -7.7876372, -0.3383553, 0.3373137
6: -4.5128145, -3.8918164, -4.5210166, -3.8959651, -0.2749739, 0.2847540
7: -8.2826242, -7.7452507, -8.2869110, -7.7421656, -0.3177633, 0.3180513
8: -1.2325611, -0.5158648, -1.2267947, -0.5115714, -0.2940755, 0.2928376
9: -7.3443165, -6.6454306, -7.3465843, -6.6452618, -0.2592442, 0.2581133

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1421323
time: 3.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
time: 3.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6441545, -6.9274426, -0.2638153, 0.2635077
1: 2.5646744, 3.0803094, 2.5663915, 3.0838852, -0.2166420, 0.2146392
2: -4.9272232, -4.3615274, -4.9274607, -4.3614988, -0.2261205, 0.2272820
3: -14.4976921, -13.4940872, -14.4957085, -13.4927607, -0.4651687, 0.4626832
4: -3.0468817, -2.4029546, -3.0466671, -2.4097390, -0.2379949, 0.2427802
5: -8.5981750, -7.7877026, -8.5985060, -7.7900581, -0.3371110, 0.3385847
6: -4.5209613, -3.8960366, -4.5131145, -3.8917797, -0.2844882, 0.2749181
7: -8.2867680, -7.7431898, -8.2827616, -7.7442083, -0.3187802, 0.3169621
8: -1.2264655, -0.5116353, -1.2326212, -0.5158162, -0.2925262, 0.2940899
9: -7.3464637, -6.6458454, -7.3443680, -6.6449294, -0.2584845, 0.2586713

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1851
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417042
time: 3.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6523972, -6.9356070, -0.2419795, 0.2419742
1: 2.5646744, 3.0803094, 2.5644341, 3.0803323, -0.2123000, 0.2125844
2: -4.9272232, -4.3615274, -4.9274716, -4.3614545, -0.2267520, 0.2269254
3: -14.4976921, -13.4940872, -14.4977264, -13.4934292, -0.4649186, 0.4642749
4: -3.0468817, -2.4029546, -3.0469060, -2.4025278, -0.2394452, 0.2390200
5: -8.5981750, -7.7877026, -8.5984735, -7.7876372, -0.3385239, 0.3387561
6: -4.5209613, -3.8960366, -4.5210166, -3.8959651, -0.2815423, 0.2815400
7: -8.2867680, -7.7431898, -8.2869110, -7.7421656, -0.3282247, 0.3274683
8: -1.2264655, -0.5116353, -1.2267947, -0.5115714, -0.2846923, 0.2849787
9: -7.3464637, -6.6458454, -7.3465843, -6.6452618, -0.2583287, 0.2579058

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1747
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
time: 3.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417042
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6441545, -6.9274426, -0.2400299, 0.2404305
1: 2.5667019, 3.0838671, 2.5663915, 3.0838852, -0.2071801, 0.2075635
2: -4.9271917, -4.3615236, -4.9274607, -4.3614988, -0.2265370, 0.2267936
3: -14.4956598, -13.4933548, -14.4957085, -13.4927607, -0.4636214, 0.4630342
4: -3.0466380, -2.4103155, -3.0466671, -2.4097390, -0.2383970, 0.2378232
5: -8.5982609, -7.7901149, -8.5985060, -7.7900581, -0.3374929, 0.3376930
6: -4.5128145, -3.8918164, -4.5131145, -3.8917797, -0.2787409, 0.2790066
7: -8.2826242, -7.7452507, -8.2827616, -7.7442083, -0.3184037, 0.3176018
8: -1.2325611, -0.5158648, -1.2326212, -0.5158162, -0.2844763, 0.2845142
9: -7.3443165, -6.6454306, -7.3443680, -6.6449294, -0.2563474, 0.2558656

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: B, layer: 3, pos: 1500
type: A, layer: 3, pos: 1851
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 409
type: A, layer: 3, pos: 409
type: B, layer: 3, pos: 1747
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2215
type: A, layer: 3, pos: 2215
type: B, layer: 3, pos: 67
type: A, layer: 3, pos: 67
type: B, layer: 3, pos: 2866
type: A, layer: 3, pos: 2866
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 233
type: B, layer: 3, pos: 233
type: A, layer: 3, pos: 2005
type: B, layer: 3, pos: 2005
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 2483
type: B, layer: 3, pos: 2483
type: A, layer: 3, pos: 915
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 1258
type: A, layer: 3, pos: 1258
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1421323
time: 3.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417040
time: 3.16 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.91 + 550.38 = 605.29 seconds
