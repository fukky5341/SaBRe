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
execution time: IAR + RelationalAnalysis = 21.78 + 32.79 = 54.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1452866, upper bound: 0.1452870

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1452856
time: 2.87 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1452855
time: 2.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.83
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1452856
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.83
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

Time for backsubstitution: 19.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1451819
time: 2.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1452855
time: 2.70 seconds

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

Time for backsubstitution: 20.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1451818
time: 2.78 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1452855
time: 2.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 25.91 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1451819
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 1, lower bound: -0.1451814, upper bound: 0.1452855
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1451818
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1452855

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.6445098, -6.9111762, -7.6445098, -6.9111762, -0.2667981, 0.2667983
1: 2.5627270, 3.0936780, 2.5627270, 3.0936780, -0.2287549, 0.2287549
2: -4.9284382, -4.3580036, -4.9284382, -4.3580036, -0.2327652, 0.2327652
3: -14.4969034, -13.4907455, -14.4969034, -13.4907455, -0.4666171, 0.4666169
4: -3.0553031, -2.4094348, -3.0553031, -2.4094348, -0.2468483, 0.2468483
5: -8.6011906, -7.7889872, -8.6011906, -7.7889872, -0.3409004, 0.3409002
6: -4.5142226, -3.8795028, -4.5142226, -3.8795028, -0.2927926, 0.2927924
7: -8.2882624, -7.7436323, -8.2882624, -7.7436323, -0.3299186, 0.3299187
8: -1.2434819, -0.5141721, -1.2434819, -0.5141721, -0.3002963, 0.3002962
9: -7.3486681, -6.6414757, -7.3486681, -6.6414757, -0.2702851, 0.2702851

Time for backsubstitution: 20.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 151

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419468
time: 3.00 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417541
time: 2.77 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.6445098, -6.9111762, -7.6452475, -6.9100342, -0.2675502, 0.2673905
1: 2.5627270, 3.0936780, 2.5614853, 3.0961642, -0.2310176, 0.2304543
2: -4.9284382, -4.3580036, -4.9295120, -4.3562870, -0.2344716, 0.2341967
3: -14.4969034, -13.4907455, -14.5021038, -13.4898701, -0.4674928, 0.4716375
4: -3.0553031, -2.4094348, -3.0601912, -2.4085894, -0.2476234, 0.2511673
5: -8.6011906, -7.7889872, -8.6015844, -7.7864594, -0.3433049, 0.3413053
6: -4.5142226, -3.8795028, -4.5146618, -3.8774104, -0.2948313, 0.2932429
7: -8.2882624, -7.7436323, -8.2967701, -7.7418885, -0.3316517, 0.3339205
8: -1.2434819, -0.5141721, -1.2443509, -0.5129433, -0.3019414, 0.3009274
9: -7.3486681, -6.6414757, -7.3521113, -6.6404891, -0.2710679, 0.2729489

Time for backsubstitution: 20.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 151

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419467
time: 3.07 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417803
time: 2.78 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.6452475, -6.9100342, -7.6445098, -6.9111762, -0.2673906, 0.2675502
1: 2.5614853, 3.0961642, 2.5627270, 3.0936780, -0.2304543, 0.2310176
2: -4.9295120, -4.3562870, -4.9284382, -4.3580036, -0.2341967, 0.2344716
3: -14.5021038, -13.4898701, -14.4969034, -13.4907455, -0.4716372, 0.4674928
4: -3.0601912, -2.4085894, -3.0553031, -2.4094348, -0.2511672, 0.2476234
5: -8.6015844, -7.7864594, -8.6011906, -7.7889872, -0.3413053, 0.3433051
6: -4.5146618, -3.8774104, -4.5142226, -3.8795028, -0.2932429, 0.2948314
7: -8.2967701, -7.7418885, -8.2882624, -7.7436323, -0.3339205, 0.3316516
8: -1.2443509, -0.5129433, -1.2434819, -0.5141721, -0.3009274, 0.3019414
9: -7.3521113, -6.6404891, -7.3486681, -6.6414757, -0.2729489, 0.2710679

Time for backsubstitution: 20.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 151

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419451
time: 3.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417526
time: 2.73 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.6452475, -6.9100342, -7.6452475, -6.9100342, -0.2682130, 0.2682130
1: 2.5614853, 3.0961642, 2.5614853, 3.0961642, -0.2319511, 0.2319511
2: -4.9295120, -4.3562870, -4.9295120, -4.3562870, -0.2345703, 0.2345703
3: -14.5021038, -13.4898701, -14.5021038, -13.4898701, -0.4688148, 0.4688146
4: -3.0601912, -2.4085894, -3.0601912, -2.4085894, -0.2489368, 0.2489367
5: -8.6015844, -7.7864594, -8.6015844, -7.7864594, -0.3424530, 0.3424530
6: -4.5146618, -3.8774104, -4.5146618, -3.8774104, -0.2942038, 0.2942036
7: -8.2967701, -7.7418885, -8.2967701, -7.7418885, -0.3336225, 0.3336225
8: -1.2443509, -0.5129433, -1.2443509, -0.5129433, -0.3025103, 0.3025105
9: -7.3521113, -6.6404891, -7.3521113, -6.6404891, -0.2718537, 0.2718537

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 151

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419451
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417716
time: 2.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 26.99 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419468
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417541
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419467
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417521, upper bound: 0.1417803
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419451
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417526
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417026, upper bound: 0.1419451
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.99
Output dim: 1, lower bound: -0.1417797, upper bound: 0.1417716

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6441450, -6.9191666, -0.2525120, 0.2403733
1: 2.5667019, 3.0838671, 2.5648222, 3.0886507, -0.2121885, 0.2113844
2: -4.9271917, -4.3615236, -4.9278183, -4.3598652, -0.2283245, 0.2276622
3: -14.4956598, -13.4933548, -14.4962807, -13.4920635, -0.4642854, 0.4633644
4: -3.0466380, -2.4103155, -3.0510044, -2.4098778, -0.2386942, 0.2410438
5: -8.5982609, -7.7901149, -8.5997419, -7.7895613, -0.3377967, 0.3386407
6: -4.5128145, -3.8918164, -4.5134807, -3.8855560, -0.2843642, 0.2796844
7: -8.2826242, -7.7452507, -8.2854271, -7.7444377, -0.3204191, 0.3202897
8: -1.2325611, -0.5158648, -1.2381215, -0.5150447, -0.2854710, 0.2907642
9: -7.3443165, -6.6454306, -7.3464046, -6.6434131, -0.2584527, 0.2600212

Time for backsubstitution: 20.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417041
time: 2.98 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417474
time: 3.02 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6442018, -6.9247236, -0.2875874, 0.2415925
1: 2.5646744, 3.0803094, 2.5650072, 3.0860300, -0.2261760, 0.2144613
2: -4.9272232, -4.3615274, -4.9275565, -4.3601975, -0.2278740, 0.2277292
3: -14.4976921, -13.4940872, -14.4963627, -13.4926376, -0.4654224, 0.4631772
4: -3.0468817, -2.4029546, -3.0504665, -2.4099293, -0.2396307, 0.2454543
5: -8.5981750, -7.7877026, -8.5995340, -7.7894578, -0.3378220, 0.3391638
6: -4.5209613, -3.8960366, -4.5135369, -3.8891869, -0.2871251, 0.2818854
7: -8.2867680, -7.7431898, -8.2868977, -7.7447681, -0.3284967, 0.3182799
8: -1.2264655, -0.5116353, -1.2341297, -0.5150685, -0.2850068, 0.3038480
9: -7.3464637, -6.6458454, -7.3463125, -6.6439095, -0.2620571, 0.2605027

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417475, upper bound: 0.1417042
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417475, upper bound: 0.1417544
time: 2.94 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6449003, -6.9177408, -0.2537829, 0.2409827
1: 2.5667019, 3.0838671, 2.5636673, 3.0913253, -0.2147583, 0.2129731
2: -4.9271917, -4.3615236, -4.9289260, -4.3581514, -0.2300328, 0.2291040
3: -14.4956598, -13.4933548, -14.5014992, -13.4912424, -0.4651065, 0.4683890
4: -3.0466380, -2.4103155, -3.0560417, -2.4090195, -0.2394787, 0.2452160
5: -8.5982609, -7.7901149, -8.6001148, -7.7870340, -0.3401923, 0.3390150
6: -4.5128145, -3.8918164, -4.5138960, -3.8832157, -0.2867113, 0.2800961
7: -8.2826242, -7.7452507, -8.2940121, -7.7426577, -0.3220916, 0.3242435
8: -1.2325611, -0.5158648, -1.2387645, -0.5138159, -0.2871184, 0.2912973
9: -7.3443165, -6.6454306, -7.3498302, -6.6424932, -0.2592265, 0.2626342

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417041
time: 3.15 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417734
time: 3.08 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6448917, -6.9242287, -0.2881430, 0.2431471
1: 2.5646744, 3.0803094, 2.5639572, 3.0874705, -0.2275218, 0.2157062
2: -4.9272232, -4.3615274, -4.9285183, -4.3586969, -0.2293653, 0.2289007
3: -14.4976921, -13.4940872, -14.5014658, -13.4918480, -0.4662163, 0.4682450
4: -3.0468817, -2.4029546, -3.0545659, -2.4091401, -0.2401665, 0.2497000
5: -8.5981750, -7.7877026, -8.5998220, -7.7869792, -0.3404474, 0.3394642
6: -4.5209613, -3.8960366, -4.5139408, -3.8884392, -0.2877734, 0.2820100
7: -8.2867680, -7.7431898, -8.2950993, -7.7431641, -0.3299606, 0.3222439
8: -1.2264655, -0.5116353, -1.2345681, -0.5139365, -0.2880945, 0.3042696
9: -7.3464637, -6.6458454, -7.3495398, -6.6430330, -0.2627718, 0.2627085

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417041
time: 2.93 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417805
time: 3.03 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.6445446, -6.9258571, -7.6441464, -6.9191465, -0.2531660, 0.2419477
1: 2.5656366, 3.0866508, 2.5648179, 3.0886631, -0.2136076, 0.2141294
2: -4.9283214, -4.3598437, -4.9278197, -4.3598604, -0.2297504, 0.2293378
3: -14.5008678, -13.4926138, -14.4962835, -13.4920597, -0.4693172, 0.4641073
4: -3.0517821, -2.4094529, -3.0510154, -2.4098773, -0.2425748, 0.2418272
5: -8.5985861, -7.7876010, -8.5997458, -7.7895608, -0.3381207, 0.3410211
6: -4.5131898, -3.8893316, -4.5134835, -3.8855386, -0.2847347, 0.2821739
7: -8.2912483, -7.7434464, -8.2854347, -7.7444358, -0.3242404, 0.3218822
8: -1.2328823, -0.5146537, -1.2381344, -0.5150418, -0.2857237, 0.2924113
9: -7.3476920, -6.6446052, -7.3464093, -6.6434102, -0.2608849, 0.2607050

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417041, upper bound: 0.1417395
time: 3.09 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417041, upper bound: 0.1417455
time: 4.26 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.6540842, -6.9353251, -7.6442089, -6.9244566, -0.2894809, 0.2421346
1: 2.5637808, 3.0812821, 2.5649738, 3.0861835, -0.2274940, 0.2154568
2: -4.9281354, -4.3600812, -4.9275732, -4.3601608, -0.2290259, 0.2291957
3: -14.5029688, -13.4932823, -14.4963703, -13.4926023, -0.4705412, 0.4639952
4: -3.0505385, -2.4022851, -3.0505633, -2.4099212, -0.2434204, 0.2461281
5: -8.5984716, -7.7849202, -8.5995684, -7.7894506, -0.3381393, 0.3418186
6: -4.5211654, -3.8957918, -4.5135460, -3.8890085, -0.2874479, 0.2822008
7: -8.2947931, -7.7415829, -8.2869234, -7.7447462, -0.3321223, 0.3198159
8: -1.2269917, -0.5086613, -1.2343161, -0.5150528, -0.2854819, 0.3071513
9: -7.3502388, -6.6449623, -7.3463516, -6.6438594, -0.2645429, 0.2613105

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417735, upper bound: 0.1417395
time: 3.06 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417735, upper bound: 0.1417526
time: 3.02 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.6445446, -6.9258571, -7.6449003, -6.9177213, -0.2538365, 0.2418461
1: 2.5656366, 3.0866508, 2.5636606, 3.0913372, -0.2153078, 0.2143711
2: -4.9283214, -4.3598437, -4.9289269, -4.3581476, -0.2300727, 0.2293837
3: -14.5008678, -13.4926138, -14.5014992, -13.4912395, -0.4665370, 0.4656060
4: -3.0517821, -2.4094529, -3.0560517, -2.4090185, -0.2410589, 0.2433207
5: -8.5985861, -7.7876010, -8.6001167, -7.7870350, -0.3394153, 0.3402138
6: -4.5131898, -3.8893316, -4.5138979, -3.8832016, -0.2855024, 0.2805035
7: -8.2912483, -7.7434464, -8.2940197, -7.7426553, -0.3239434, 0.3240355
8: -1.2328823, -0.5146537, -1.2387791, -0.5138140, -0.2873089, 0.2928818
9: -7.3476920, -6.6446052, -7.3498373, -6.6424856, -0.2599455, 0.2615101

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417116, upper bound: 0.1417466
time: 3.01 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417116, upper bound: 0.1417555
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.6540842, -6.9353251, -7.6448975, -6.9239645, -0.2909443, 0.2428941
1: 2.5637808, 3.0812821, 2.5639229, 3.0876360, -0.2302759, 0.2165247
2: -4.9281354, -4.3600812, -4.9285374, -4.3586593, -0.2294254, 0.2291565
3: -14.5029688, -13.4932823, -14.5014791, -13.4918118, -0.4673290, 0.4651434
4: -3.0505385, -2.4022851, -3.0546718, -2.4091311, -0.2415506, 0.2479806
5: -8.5984716, -7.7849202, -8.5998554, -7.7869706, -0.3392854, 0.3404217
6: -4.5211654, -3.8957918, -4.5139494, -3.8882444, -0.2903314, 0.2824765
7: -8.2947931, -7.7415829, -8.2951279, -7.7431402, -0.3313570, 0.3220383
8: -1.2269917, -0.5086613, -1.2347541, -0.5139198, -0.2869530, 0.3075749
9: -7.3502388, -6.6449623, -7.3495789, -6.6429839, -0.2638435, 0.2617260

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1500
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 409
type: B, layer: 3, pos: 1851
type: B, layer: 3, pos: 2215
type: B, layer: 3, pos: 1747
type: B, layer: 3, pos: 2866
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 67
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 2005
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 1258
type: B, layer: 3, pos: 2483
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 915
type: B, layer: 3, pos: 151

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1102

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417734, upper bound: 0.1417459
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417718
time: 3.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.98 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417041
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417474
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417475, upper bound: 0.1417042
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417475, upper bound: 0.1417544
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417041
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417734
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417041
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417458, upper bound: 0.1417805
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417041, upper bound: 0.1417395
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417041, upper bound: 0.1417455
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417735, upper bound: 0.1417395
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417735, upper bound: 0.1417526
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417116, upper bound: 0.1417466
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417116, upper bound: 0.1417555
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417734, upper bound: 0.1417459
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.98
Output dim: 1, lower bound: -0.1417733, upper bound: 0.1417718

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6437759, -6.9274640, -0.2400044, 0.2400044
1: 2.5667019, 3.0838671, 2.5667019, 3.0838671, -0.2071313, 0.2071312
2: -4.9271917, -4.3615236, -4.9271917, -4.3615236, -0.2265081, 0.2265081
3: -14.4956598, -13.4933548, -14.4956598, -13.4933548, -0.4629116, 0.4629116
4: -3.0466380, -2.4103155, -3.0466380, -2.4103155, -0.2378012, 0.2378012
5: -8.5982609, -7.7901149, -8.5982609, -7.7901149, -0.3373351, 0.3373351
6: -4.5128145, -3.8918164, -4.5128145, -3.8918164, -0.2786906, 0.2786906
7: -8.2826242, -7.7452507, -8.2826242, -7.7452507, -0.3172188, 0.3172189
8: -1.2325611, -0.5158648, -1.2325611, -0.5158648, -0.2844458, 0.2844455
9: -7.3443165, -6.6454306, -7.3443165, -6.6454306, -0.2557935, 0.2557935

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 151

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1419469
time: 3.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417043
time: 3.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6522465, -6.9358864, -0.2630818, 0.2637897
1: 2.5667019, 3.0838671, 2.5646744, 3.0803094, -0.2142069, 0.2165931
2: -4.9271917, -4.3615236, -4.9272232, -4.3615274, -0.2269967, 0.2260917
3: -14.4956598, -13.4933548, -14.4976921, -13.4940872, -0.4625609, 0.4644589
4: -3.0466380, -2.4103155, -3.0468817, -2.4029546, -0.2427583, 0.2373991
5: -8.5982609, -7.7901149, -8.5981750, -7.7877026, -0.3382268, 0.3369527
6: -4.5128145, -3.8918164, -4.5209613, -3.8960366, -0.2746019, 0.2844378
7: -8.2826242, -7.7452507, -8.2867680, -7.7431898, -0.3165793, 0.3175952
8: -1.2325611, -0.5158648, -1.2264655, -0.5116353, -0.2940214, 0.2924954
9: -7.3443165, -6.6454306, -7.3464637, -6.6458454, -0.2585992, 0.2579305

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 151

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1419468
time: 3.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417475
time: 2.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6437759, -6.9274640, -0.2637898, 0.2630817
1: 2.5646744, 3.0803094, 2.5667019, 3.0838671, -0.2165931, 0.2142069
2: -4.9272232, -4.3615274, -4.9271917, -4.3615236, -0.2260917, 0.2269965
3: -14.4976921, -13.4940872, -14.4956598, -13.4933548, -0.4644589, 0.4625609
4: -3.0468817, -2.4029546, -3.0466380, -2.4103155, -0.2373992, 0.2427582
5: -8.5981750, -7.7877026, -8.5982609, -7.7901149, -0.3369529, 0.3382268
6: -4.5209613, -3.8960366, -4.5128145, -3.8918164, -0.2844379, 0.2746019
7: -8.2867680, -7.7431898, -8.2826242, -7.7452507, -0.3175950, 0.3165792
8: -1.2264655, -0.5116353, -1.2325611, -0.5158648, -0.2924953, 0.2940214
9: -7.3464637, -6.6458454, -7.3443165, -6.6454306, -0.2579305, 0.2585992

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 151

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417043
time: 3.16 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417475, upper bound: 0.1417043
time: 3.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.6522465, -6.9358864, -7.6522465, -6.9358864, -0.2416629, 0.2416628
1: 2.5646744, 3.0803094, 2.5646744, 3.0803094, -0.2122695, 0.2122695
2: -4.9272232, -4.3615274, -4.9272232, -4.3615274, -0.2266641, 0.2266641
3: -14.4976921, -13.4940872, -14.4976921, -13.4940872, -0.4641750, 0.4641750
4: -3.0468817, -2.4029546, -3.0468817, -2.4029546, -0.2390020, 0.2390021
5: -8.5981750, -7.7877026, -8.5981750, -7.7877026, -0.3383613, 0.3383613
6: -4.5209613, -3.8960366, -4.5209613, -3.8960366, -0.2814474, 0.2814475
7: -8.2867680, -7.7431898, -8.2867680, -7.7431898, -0.3271515, 0.3271517
8: -1.2264655, -0.5116353, -1.2264655, -0.5116353, -0.2846384, 0.2846384
9: -7.3464637, -6.6458454, -7.3464637, -6.6458454, -0.2576917, 0.2576917

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 151

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417042, upper bound: 0.1417475
time: 3.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417538, upper bound: 0.1417542
time: 2.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6445446, -6.9258571, -0.2415777, 0.2406249
1: 2.5667019, 3.0838671, 2.5656366, 3.0866508, -0.2098640, 0.2085326
2: -4.9271917, -4.3615236, -4.9283214, -4.3598437, -0.2281806, 0.2279284
3: -14.4956598, -13.4933548, -14.5008678, -13.4926138, -0.4636531, 0.4679394
4: -3.0466380, -2.4103155, -3.0517821, -2.4094529, -0.2385759, 0.2416792
5: -8.5982609, -7.7901149, -8.5985861, -7.7876010, -0.3397117, 0.3376579
6: -4.5128145, -3.8918164, -4.5131898, -3.8893316, -0.2811773, 0.2790467
7: -8.2826242, -7.7452507, -8.2912483, -7.7434464, -0.3188028, 0.3210499
8: -1.2325611, -0.5158648, -1.2328823, -0.5146537, -0.2860754, 0.2846955
9: -7.3443165, -6.6454306, -7.3476920, -6.6446052, -0.2564651, 0.2581954

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1500
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 409
type: A, layer: 3, pos: 1851
type: A, layer: 3, pos: 2215
type: A, layer: 3, pos: 1747
type: A, layer: 3, pos: 2866
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 67
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2005
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1258
type: A, layer: 3, pos: 2483
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 915
type: A, layer: 3, pos: 151

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 1102

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1421322
time: 3.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417396, upper bound: 0.1417041
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.6437759, -6.9274640, -7.6540842, -6.9353251, -0.2636497, 0.2656546
1: 2.5667019, 3.0838671, 2.5637808, 3.0812821, -0.2151656, 0.2178570
2: -4.9271917, -4.3615236, -4.9281354, -4.3600812, -0.2284335, 0.2272083
3: -14.4956598, -13.4933548, -14.5029688, -13.4932823, -0.4633498, 0.4695449
4: -3.0466380, -2.4103155, -3.0505385, -2.4022851, -0.2433491, 0.2411797
5: -8.5982609, -7.7901149, -8.5984716, -7.7849202, -0.3408496, 0.3372614
6: -4.5128145, -3.8918164, -4.5211654, -3.8957918, -0.2748358, 0.2845763
7: -8.2826242, -7.7452507, -8.2947931, -7.7415829, -0.3180351, 0.3214499
8: -1.2325611, -0.5158648, -1.2269917, -0.5086613, -0.2971417, 0.2929465
9: -7.3443165, -6.6454306, -7.3502388, -6.6449623, -0.2593002, 0.2604282

Time for backsubstitution: 21.39 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.57 + 548.50 = 603.07 seconds
