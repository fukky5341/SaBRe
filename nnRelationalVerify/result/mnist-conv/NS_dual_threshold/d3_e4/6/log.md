## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1823463684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5293298, 2.5293295)
1: (-10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2577815, 2.2577815)
2: (-6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3718305, 2.3718295)
3: (-2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8441834, 1.8441832)
4: (-6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1593237, 3.1593237)
5: (-8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4321971, 2.4321966)
6: (-19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1931105, 3.1931105)
7: (4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772)
8: (-7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3909245, 2.3909245)
9: (-7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6847959, 2.6847959)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.99 + 33.71 = 58.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.1847179, upper bound: 1.1847154

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 457
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 457

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1841504, upper bound: 1.1791575
time: 7.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847100, upper bound: 1.1847084
time: 4.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.68
Output dim: 7, lower bound: -1.1841504, upper bound: 1.1791575
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.68
Output dim: 7, lower bound: -1.1847100, upper bound: 1.1847084

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -17.5882187, -13.5900822, -17.5930214, -13.5878315, -2.5169849, 2.5178177
1: -10.2623758, -7.4767728, -10.2640038, -7.4714136, -2.2490282, 2.2455969
2: -6.4378533, -3.5996528, -6.4474549, -3.5983841, -2.3524609, 2.3608422
3: -2.4340112, 0.1182401, -2.4360194, 0.1221886, -1.8353758, 1.8326530
4: -6.9883199, -2.9186769, -6.9913054, -2.9069707, -3.1420994, 3.1336880
5: -8.9537373, -5.7457619, -8.9571953, -5.7410479, -2.4210677, 2.4190965
6: -19.4427872, -15.5619993, -19.4446411, -15.5569839, -3.1816673, 3.1770735
7: 4.2643223, 6.9667130, 4.2619171, 6.9752622, -2.7109399, 2.7047958
8: -7.1617842, -4.4029832, -7.1654897, -4.4018068, -2.3793960, 2.3808310
9: -7.2016177, -3.7783475, -7.2060957, -3.7777143, -2.6713328, 2.6756620

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791576, upper bound: 1.1791555
time: 4.41 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791554, upper bound: 1.1791547
time: 4.47 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -17.6044273, -13.5805111, -17.5972481, -13.5857992, -2.5366182, 2.5315771
1: -10.2822809, -7.4614162, -10.2654266, -7.4666910, -2.2746010, 2.2608061
2: -6.4601746, -3.5581911, -6.4559097, -3.5972750, -2.3724961, 2.3958046
3: -2.4422810, 0.1332535, -2.4377654, 0.1256831, -1.8518305, 1.8501282
4: -7.0440598, -2.8905506, -6.9938722, -2.8966660, -3.1867456, 3.1572790
5: -8.9876623, -5.7355223, -8.9602032, -5.7369003, -2.4638071, 2.4334931
6: -19.4601669, -15.5480824, -19.4462547, -15.5525627, -3.2123308, 3.1964045
7: 4.2270660, 6.9874487, 4.2598314, 6.9827952, -2.7557292, 2.7276173
8: -7.1751170, -4.3977690, -7.1687737, -4.4007764, -2.3974919, 2.3905525
9: -7.2168636, -3.7630317, -7.2100434, -3.7771645, -2.6893601, 2.7021294

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6209
type: B, layer: 1, pos: 6209
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6209

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1816141, upper bound: 1.1791819
time: 4.35 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847088, upper bound: 1.1847069
time: 4.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.38 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.38
Output dim: 7, lower bound: -1.1791576, upper bound: 1.1791555
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 31.38
Output dim: 7, lower bound: -1.1791554, upper bound: 1.1791547
NS_A2_A1, status: Status.VERIFIED, split count: 2, time: 31.38
Output dim: 7, lower bound: -1.1816141, upper bound: 1.1791819
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 31.38
Output dim: 7, lower bound: -1.1847088, upper bound: 1.1847069

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -17.6044273, -13.5805216, -17.5972481, -13.5858002, -2.5366154, 2.4823136
1: -10.2822809, -7.4614220, -10.2654285, -7.4666910, -2.2746005, 2.2344270
2: -6.4601727, -3.5581913, -6.4559102, -3.5972743, -2.3207345, 2.3862245
3: -2.4422774, 0.1332517, -2.4377644, 0.1256831, -1.8223114, 1.8501267
4: -7.0440578, -2.8905520, -6.9938722, -2.8966637, -3.1829033, 3.1401196
5: -8.9876575, -5.7355213, -8.9602051, -5.7369003, -2.4454947, 2.4334927
6: -19.4601669, -15.5480900, -19.4462528, -15.5525627, -3.2123299, 3.1675348
7: 4.2270679, 6.9874458, 4.2598319, 6.9827938, -2.7557259, 2.7276139
8: -7.1751137, -4.3977704, -7.1687727, -4.4007759, -2.3974900, 2.3876011
9: -7.2168627, -3.7630353, -7.2100453, -3.7771645, -2.6891060, 2.6797829

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 478
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 478

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847077, upper bound: 1.1825597
time: 5.03 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847077, upper bound: 1.1847058
time: 4.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.65 seconds
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.65
Output dim: 7, lower bound: -1.1847077, upper bound: 1.1825597
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.65
Output dim: 7, lower bound: -1.1847077, upper bound: 1.1847058

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -17.5992374, -13.5838604, -17.5848026, -13.5906773, -2.5259869, 2.4660668
1: -10.2801304, -7.4985852, -10.2407970, -7.5379014, -2.2014403, 2.1728420
2: -6.4147611, -3.5602384, -6.3695860, -3.6265092, -2.2385674, 2.2976520
3: -2.4262710, 0.1311426, -2.4066987, 0.1139750, -1.7944093, 1.8172197
4: -7.0415039, -2.9192042, -6.9735928, -2.9521716, -3.1246824, 3.0910077
5: -8.9848452, -5.7430944, -8.9478693, -5.7516851, -2.4277411, 2.4110456
6: -19.4459820, -15.5499029, -19.4181862, -15.5658503, -3.1758566, 3.1383600
7: 4.2337809, 6.9838891, 4.2735128, 6.9688668, -2.7350860, 2.7103763
8: -7.1703568, -4.4277201, -7.1420617, -4.4574785, -2.3355875, 2.3186100
9: -7.2119184, -3.7737806, -7.1884136, -3.7976499, -2.6622591, 2.6369810

Time for backsubstitution: 22.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809623, upper bound: 1.1748080
time: 5.45 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847055, upper bound: 1.1825572
time: 6.85 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805225, -17.5972462, -13.5858002, -2.5366144, 2.4810767
1: -10.2822781, -7.4614220, -10.2654266, -7.4666929, -2.2241406, 2.2344260
2: -6.4601722, -3.5581913, -6.4559059, -3.5972743, -2.3135223, 2.3020871
3: -2.4422770, 0.1332513, -2.4377632, 0.1256844, -1.8223104, 1.8318918
4: -7.0440588, -2.8905535, -6.9938707, -2.8966680, -3.1398268, 3.1401196
5: -8.9876595, -5.7355232, -8.9602032, -5.7369013, -2.4330163, 2.4334893
6: -19.4601650, -15.5480928, -19.4462452, -15.5525627, -3.2092600, 3.1553898
7: 4.2270679, 6.9874463, 4.2598338, 6.9827919, -2.7557240, 2.7276125
8: -7.1751146, -4.3977733, -7.1687727, -4.4007788, -2.3481812, 2.3840568
9: -7.2168651, -3.7630353, -7.2100453, -3.7771659, -2.6916752, 2.6739750

Time for backsubstitution: 22.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847024, upper bound: 1.1833961
time: 4.55 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847004, upper bound: 1.1846981
time: 4.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.27 seconds
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 32.27
Output dim: 7, lower bound: -1.1809623, upper bound: 1.1748080
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 32.27
Output dim: 7, lower bound: -1.1847055, upper bound: 1.1825572
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.27
Output dim: 7, lower bound: -1.1847024, upper bound: 1.1833961
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.27
Output dim: 7, lower bound: -1.1847004, upper bound: 1.1846981

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.5992393, -13.5838623, -17.5848026, -13.5906773, -2.5237029, 2.4563344
1: -10.2801294, -7.4985862, -10.2407970, -7.5379014, -2.2014389, 2.1754208
2: -6.4147592, -3.5602388, -6.3695860, -3.6265092, -2.2245016, 2.2946663
3: -2.4262688, 0.1311424, -2.4066987, 0.1139750, -1.7519021, 1.8172204
4: -7.0415025, -2.9192057, -6.9735928, -2.9521716, -3.1229992, 3.0833545
5: -8.9848404, -5.7430954, -8.9478693, -5.7516851, -2.3836846, 2.4110451
6: -19.4459820, -15.5499001, -19.4181862, -15.5658503, -3.1758566, 3.1167507
7: 4.2337828, 6.9838881, 4.2735128, 6.9688668, -2.7350841, 2.7103753
8: -7.1703558, -4.4277225, -7.1420617, -4.4574785, -2.3352356, 2.2974818
9: -7.2119179, -3.7737820, -7.1884136, -3.7976499, -2.6622596, 2.6316972

Time for backsubstitution: 23.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5746

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1812814
time: 4.54 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846982, upper bound: 1.1825495
time: 4.87 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805225, -17.5951481, -13.5864277, -2.5338945, 2.4773226
1: -10.2822781, -7.4614220, -10.2611771, -7.4682851, -2.2225714, 2.2300854
2: -6.4601722, -3.5581913, -6.4552727, -3.6010270, -2.3097382, 2.3013709
3: -2.4422770, 0.1332513, -2.4361448, 0.1203659, -1.8172898, 1.8304117
4: -7.0440588, -2.8905535, -6.9880161, -2.8980763, -3.1383905, 3.1343508
5: -8.9876595, -5.7355232, -8.9580593, -5.7373953, -2.4317379, 2.4308534
6: -19.4601650, -15.5480928, -19.4452133, -15.5538998, -3.2070608, 3.1535521
7: 4.2270679, 6.9874463, 4.2631283, 6.9818120, -2.7547441, 2.7243180
8: -7.1751146, -4.3977733, -7.1660690, -4.4038391, -2.3450861, 2.3813028
9: -7.2168651, -3.7630353, -7.2067633, -3.7801445, -2.6864581, 2.6688838

Time for backsubstitution: 23.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809552, upper bound: 1.1700408
time: 5.68 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1833931
time: 4.40 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17.6044121, -13.5805254, -17.6048145, -13.5801516, -2.5413527, 2.5046303
1: -10.2822571, -7.4614282, -10.2709188, -7.4142847, -2.2411785, 2.2407722
2: -6.4601688, -3.5582023, -6.4833503, -3.5910387, -2.3198791, 2.3084698
3: -2.4422696, 0.1332281, -2.4767208, 0.1330500, -1.8321338, 1.8438585
4: -7.0440283, -2.8905585, -7.0033517, -2.8442810, -3.1546369, 3.1496382
5: -8.9876499, -5.7355223, -8.9705763, -5.7332087, -2.4401951, 2.4446115
6: -19.4601612, -15.5480938, -19.4527397, -15.5467005, -3.2225189, 3.1612272
7: 4.2270789, 6.9874430, 4.2460842, 7.0008888, -2.7738099, 2.7413588
8: -7.1751051, -4.3977847, -7.2139788, -4.3977222, -2.3520226, 2.4030049
9: -7.2168512, -3.7630455, -7.2608643, -3.7744656, -2.6923118, 2.7161112

Time for backsubstitution: 23.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 539

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1809550, upper bound: 1.1769447
time: 5.90 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846982, upper bound: 1.1846951
time: 4.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.44 seconds
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1812814
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 7, lower bound: -1.1846982, upper bound: 1.1825495
NS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 34.44
Output dim: 7, lower bound: -1.1809552, upper bound: 1.1700408
NS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 7, lower bound: -1.1846985, upper bound: 1.1833931
NS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.44
Output dim: 7, lower bound: -1.1809550, upper bound: 1.1769447
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 7, lower bound: -1.1846982, upper bound: 1.1846951

## BFS NS instance: NS_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.5992393, -13.5838623, -17.5826874, -13.5913153, -2.5209737, 2.4525752
1: -10.2801294, -7.4985862, -10.2365503, -7.5395069, -2.1998568, 2.1710792
2: -6.4147592, -3.5602388, -6.3689470, -3.6302609, -2.2207170, 2.2939539
3: -2.4262688, 0.1311424, -2.4050670, 0.1086574, -1.7468791, 1.8157649
4: -7.0415025, -2.9192057, -6.9677358, -2.9535930, -3.1215553, 3.0775867
5: -8.9848404, -5.7430954, -8.9457321, -5.7521791, -2.3824039, 2.4084272
6: -19.4459820, -15.5499001, -19.4171543, -15.5671721, -3.1736774, 3.1149135
7: 4.2337828, 6.9838881, 4.2768207, 6.9679022, -2.7341194, 2.7070675
8: -7.1703558, -4.4277225, -7.1393681, -4.4605379, -2.3321424, 2.2947230
9: -7.2119179, -3.7737820, -7.1851988, -3.8006284, -2.6570425, 2.6266909

Time for backsubstitution: 23.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of NS_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833954, upper bound: 1.1812816
time: 5.01 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833954, upper bound: 1.1812813
time: 4.70 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.5992279, -13.5838642, -17.5922775, -13.5850077, -2.5246835, 2.4807081
1: -10.2801075, -7.4985905, -10.2463369, -7.4856544, -2.2215283, 2.1818042
2: -6.4147568, -3.5602489, -6.3969307, -3.6202116, -2.2308989, 2.3010497
3: -2.4262602, 0.1311200, -2.4456921, 0.1214027, -1.7617147, 1.8252485
4: -7.0414720, -2.9192123, -6.9830837, -2.8998809, -3.1378136, 3.0928860
5: -8.9848328, -5.7430973, -8.9583788, -5.7479801, -2.3918343, 2.4222665
6: -19.4459763, -15.5499058, -19.4246864, -15.5598183, -3.1899061, 3.1225796
7: 4.2337933, 6.9838848, 4.2598162, 6.9871340, -2.7533407, 2.7240686
8: -7.1703472, -4.4277349, -7.1874390, -4.4544187, -2.3390861, 2.3235962
9: -7.2119074, -3.7737899, -7.2395601, -3.7949538, -2.6628981, 2.6882372

Time for backsubstitution: 23.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791452, upper bound: 1.1819892
time: 4.60 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791430, upper bound: 1.1825504
time: 4.82 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -17.6044235, -13.5805225, -17.5951481, -13.5864277, -2.5331097, 2.4675858
1: -10.2822790, -7.4614224, -10.2611771, -7.4682851, -2.2225704, 2.2326646
2: -6.4601698, -3.5581932, -6.4552727, -3.6010270, -2.2956715, 2.2983830
3: -2.4422712, 0.1332488, -2.4361448, 0.1203659, -1.7747836, 1.8304102
4: -7.0440578, -2.8905537, -6.9880161, -2.8980763, -3.1367092, 3.1266966
5: -8.9876537, -5.7355242, -8.9580593, -5.7373953, -2.3876791, 2.4308524
6: -19.4601631, -15.5480919, -19.4452133, -15.5538998, -3.2070599, 3.1319408
7: 4.2270679, 6.9874430, 4.2631283, 6.9818120, -2.7547441, 2.7243147
8: -7.1751146, -4.3977747, -7.1660690, -4.4038391, -2.3447332, 2.3601775
9: -7.2168617, -3.7630377, -7.2067633, -3.7801445, -2.6864581, 2.6635981

Time for backsubstitution: 23.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of NS_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833974, upper bound: 1.1833925
time: 4.49 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1833974, upper bound: 1.1833928
time: 4.95 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -17.6044159, -13.5805283, -17.6048145, -13.5801516, -2.5390682, 2.4945028
1: -10.2822571, -7.4614282, -10.2709188, -7.4142847, -2.2411780, 2.2433524
2: -6.4601660, -3.5582027, -6.4833503, -3.5910387, -2.3058124, 2.3054814
3: -2.4422648, 0.1332276, -2.4767208, 0.1330500, -1.7896271, 1.8389795
4: -7.0440283, -2.8905606, -7.0033517, -2.8442810, -3.1529527, 3.1419864
5: -8.9876480, -5.7355242, -8.9705763, -5.7332087, -2.3961005, 2.4446115
6: -19.4601593, -15.5480986, -19.4527397, -15.5467005, -3.2191525, 3.1396160
7: 4.2270803, 6.9874396, 4.2460842, 7.0008888, -2.7738085, 2.7413554
8: -7.1751046, -4.3977861, -7.2139788, -4.3977222, -2.3516693, 2.3816309
9: -7.2168484, -3.7630463, -7.2608643, -3.7744656, -2.6923108, 2.7108183

Time for backsubstitution: 24.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791430, upper bound: 1.1841352
time: 4.68 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791430, upper bound: 1.1846961
time: 4.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 33.69 seconds
NS_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1833954, upper bound: 1.1812816
NS_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1833954, upper bound: 1.1812813
NS_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1791452, upper bound: 1.1819892
NS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1791430, upper bound: 1.1825504
NS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1833974, upper bound: 1.1833925
NS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1833974, upper bound: 1.1833928
NS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1791430, upper bound: 1.1841352
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.69
Output dim: 7, lower bound: -1.1791430, upper bound: 1.1846961

## BFS NS instance: NS_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17.5971317, -13.5844955, -17.5826874, -13.5913153, -2.5172195, 2.4498591
1: -10.2758789, -7.5001841, -10.2365503, -7.5395069, -2.1955118, 2.1695037
2: -6.4141302, -3.5639973, -6.3689470, -3.6302609, -2.2199945, 2.2901645
3: -2.4246588, 0.1258248, -2.4050670, 0.1086574, -1.7454534, 1.8107393
4: -7.0356207, -2.9206390, -6.9677358, -2.9535930, -3.1157875, 3.0761447
5: -8.9826689, -5.7435904, -8.9457321, -5.7521791, -2.3797832, 2.4071431
6: -19.4449501, -15.5512438, -19.4171543, -15.5671721, -3.1718388, 3.1127257
7: 4.2370529, 6.9829168, 4.2768207, 6.9679022, -2.7308493, 2.7060962
8: -7.1676621, -4.4307823, -7.1393681, -4.4605379, -2.3293905, 2.2916288
9: -7.2086754, -3.7767677, -7.1851988, -3.8006284, -2.6520042, 2.6214666

Time for backsubstitution: 25.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 478
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 457

## Relational analysis of NS_A2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778390, upper bound: 1.1807228
time: 5.53 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778409, upper bound: 1.1812853
time: 8.07 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17.6067963, -13.5781965, -17.5826874, -13.5913153, -2.5280399, 2.4613690
1: -10.2855730, -7.4476948, -10.2365503, -7.5395069, -2.2060475, 2.1901188
2: -6.4417691, -3.5541296, -6.3689470, -3.6302609, -2.2268481, 2.3001068
3: -2.4651155, 0.1384751, -2.4050670, 0.1086574, -1.7532282, 1.8249273
4: -7.0508595, -2.8681073, -6.9677358, -2.9535930, -3.1311092, 3.0955763
5: -8.9949856, -5.7394075, -8.9457321, -5.7521791, -2.3905692, 2.4160700
6: -19.4523220, -15.5440960, -19.4171543, -15.5671721, -3.1792164, 3.1197181
7: 4.2207065, 7.0019026, 4.2768207, 6.9679022, -2.7471957, 2.7250819
8: -7.2138476, -4.4246392, -7.1393681, -4.4605379, -2.3594279, 2.2984087
9: -7.2603521, -3.7711415, -7.1851988, -3.8006284, -2.7086911, 2.6272879

Time for backsubstitution: 25.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 457
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 478

## Relational analysis of NS_A2_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1814976, upper bound: 1.1812818
time: 5.02 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1814976, upper bound: 1.1812818
time: 5.13 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17.5992279, -13.5838642, -17.5995598, -13.5797119, -2.5280881, 2.4898643
1: -10.2801075, -7.4985905, -10.2631798, -7.4804091, -2.2266495, 2.1891198
2: -6.4147568, -3.5602489, -6.4011965, -3.5811224, -2.2374315, 2.3052759
3: -2.4262602, 0.1311200, -2.4503555, 0.1290560, -1.7690115, 1.8299344
4: -7.0414720, -2.9192123, -7.0332484, -2.8938377, -3.1438470, 3.1050754
5: -8.9848328, -5.7430973, -8.9857969, -5.7466049, -2.3932486, 2.4459088
6: -19.4459763, -15.5499058, -19.4386082, -15.5553818, -3.1954298, 3.1441422
7: 4.2337933, 6.9838848, 4.2271457, 6.9917612, -2.7579679, 2.7567391
8: -7.1703472, -4.4277349, -7.1937051, -4.4513893, -2.3425226, 2.3336220
9: -7.2119074, -3.7737899, -7.2462721, -3.7808039, -2.6820388, 2.6944818

Time for backsubstitution: 25.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 478
type: B, layer: 1, pos: 6209
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5746

## Relational analysis of NS_A2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1778369, upper bound: 1.1825514
time: 6.01 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1778369, upper bound: 1.1816285
time: 4.90 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -17.6023293, -13.5811520, -17.5951481, -13.5864277, -2.5293627, 2.4648697
1: -10.2780275, -7.4630198, -10.2611771, -7.4682851, -2.2182255, 2.2310772
2: -6.4595399, -3.5619490, -6.4552727, -3.6010270, -2.2949581, 2.2945938
3: -2.4406681, 0.1279335, -2.4361448, 0.1203659, -1.7733593, 1.8253865
4: -7.0381794, -2.8919849, -6.9880161, -2.8980763, -3.1309414, 3.1252542
5: -8.9854851, -5.7360163, -8.9580593, -5.7373953, -2.3850517, 2.4295702
6: -19.4591331, -15.5494366, -19.4452133, -15.5538998, -3.2052202, 3.1297665
7: 4.2303267, 6.9864655, 4.2631283, 6.9818120, -2.7514853, 2.7233372
8: -7.1724129, -4.4008341, -7.1660690, -4.4038391, -2.3420019, 2.3570838
9: -7.2135859, -3.7660210, -7.2067633, -3.7801445, -2.6814609, 2.6583757

Time for backsubstitution: 25.75 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.71 + 543.68 = 602.38 seconds
