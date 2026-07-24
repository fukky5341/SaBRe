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
execution time: IAR + RelationalAnalysis = 23.62 + 33.10 = 56.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.1452866, upper bound: 0.1452870

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4598

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451890, upper bound: 0.1452854
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1451894
time: 2.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.55 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.55
Output dim: 1, lower bound: -0.1451890, upper bound: 0.1452854
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.55
Output dim: 1, lower bound: -0.1452850, upper bound: 0.1451894

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2674878, 0.2675886
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2292247, 0.2295976
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2329627, 0.2331822
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4675524, 0.4668515
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2476076, 0.2470753
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3410981, 0.3414826
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2931640, 0.2934581
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3317630, 0.3304993
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006511, 0.3006600
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2711654, 0.2707733

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2866

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1451786, upper bound: 0.1447053
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1446033, upper bound: 0.1452750
time: 2.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2675802, 0.2674878
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2295630, 0.2292247
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2331604, 0.2329627
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4668515, 0.4674888
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2470753, 0.2475598
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3414524, 0.3410981
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2934346, 0.2931639
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3304994, 0.3316460
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006594, 0.3006511
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2707733, 0.2711308

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1428470, upper bound: 0.1435711
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1436702, upper bound: 0.1427477
time: 2.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.14 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 1, lower bound: -0.1451786, upper bound: 0.1447053
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 1, lower bound: -0.1446033, upper bound: 0.1452750
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 1, lower bound: -0.1428470, upper bound: 0.1435711
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.14
Output dim: 1, lower bound: -0.1436702, upper bound: 0.1427477

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2607968, 0.2609975
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2276706, 0.2279161
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2164232, 0.2170126
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4545460, 0.4548724
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2469865, 0.2463522
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3415680, 0.3419313
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2694719, 0.2686828
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3226051, 0.3211737
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2668011, 0.2703060
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2635748, 0.2635254

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 424

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1447194, upper bound: 0.1443304
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1448037, upper bound: 0.1442461
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2608967, 0.2608922
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2275421, 0.2280436
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2167932, 0.2166356
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4555731, 0.4538395
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2468842, 0.2464542
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3415470, 0.3419518
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2683885, 0.2697544
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3224311, 0.3213409
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2702770, 0.2668101
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2639127, 0.2631828

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2483

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1442820, upper bound: 0.1445202
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1438485, upper bound: 0.1449539
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2620506, 0.2631072
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2281653, 0.2282174
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2323533, 0.2312555
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4664688, 0.4666030
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2452977, 0.2463226
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3299849, 0.3343565
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2932117, 0.2931166
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3297718, 0.3305442
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2968380, 0.2954677
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2615910, 0.2591366

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 424

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1420056, upper bound: 0.1422697
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1419433, upper bound: 0.1427899
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2631996, 0.2619582
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2285558, 0.2278268
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2314533, 0.2321556
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4659657, 0.4671061
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2458382, 0.2457823
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3347113, 0.3296301
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2933872, 0.2929409
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3293970, 0.3309190
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2954762, 0.2968298
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2587792, 0.2619485

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2342

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 915

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1435194, upper bound: 0.1424950
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1434164, upper bound: 0.1425980
time: 2.78 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1447194, upper bound: 0.1443304
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1448037, upper bound: 0.1442461
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1442820, upper bound: 0.1445202
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1438485, upper bound: 0.1449539
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1420056, upper bound: 0.1422697
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1419433, upper bound: 0.1427899
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1435194, upper bound: 0.1424950
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.48
Output dim: 1, lower bound: -0.1434164, upper bound: 0.1425980

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2669040, 0.2668666
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2254210, 0.2260292
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2330276, 0.2332319
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4674032, 0.4667799
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2452776, 0.2448242
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3397696, 0.3402870
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2939854, 0.2942179
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3313732, 0.3301182
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964649, 0.2969441
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2663684, 0.2656038

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1258

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1851

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1441838, upper bound: 0.1438872
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1442772, upper bound: 0.1437912
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2667657, 0.2670047
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2256563, 0.2257938
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2330124, 0.2332472
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4674807, 0.4667025
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2453564, 0.2447454
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3399026, 0.3401539
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2939239, 0.2942796
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3313820, 0.3301092
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2969353, 0.2964737
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2659959, 0.2659764

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1839

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1258

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1439576, upper bound: 0.1420913
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1439061, upper bound: 0.1434427
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2674876, 0.2675886
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2291930, 0.2295464
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2321821, 0.2324706
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4675360, 0.4668510
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2475089, 0.2471333
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3405664, 0.3406696
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2925340, 0.2930453
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3314130, 0.3304294
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006246, 0.3006366
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2702863, 0.2697192

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 409

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 424

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1437782, upper bound: 0.1436493
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1438324, upper bound: 0.1443059
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2674879, 0.2675884
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2291734, 0.2295658
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2322513, 0.2324016
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4675517, 0.4668353
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2476655, 0.2469766
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3402853, 0.3409507
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2927510, 0.2928281
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3316934, 0.3301488
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3006277, 0.3006334
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2701113, 0.2698942

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1500

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 424

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1436342, upper bound: 0.1445037
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1429776, upper bound: 0.1444500
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2634420, 0.2631390
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2291867, 0.2286901
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2306113, 0.2299469
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4579263, 0.4592168
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2458727, 0.2460163
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3360016, 0.3352528
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2906258, 0.2906771
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3212590, 0.3230929
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2960970, 0.2960142
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2667173, 0.2679397

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1258

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2005

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1415937, upper bound: 0.1412566
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1409918, upper bound: 0.1418592
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2632314, 0.2633498
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2290286, 0.2288482
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2301445, 0.2304137
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4585795, 0.4585636
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2455319, 0.2463572
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3356078, 0.3356466
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2909477, 0.2903548
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3219461, 0.3224058
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2960222, 0.2960891
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2675822, 0.2670748

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1851

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 180

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1409170, upper bound: 0.1427878
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1419415, upper bound: 0.1417621
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2651969, 0.2655975
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2294158, 0.2290688
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2203747, 0.2218019
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4689834, 0.4689691
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2506702, 0.2508988
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3395805, 0.3391080
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2941113, 0.2939262
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3287251, 0.3302071
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3050799, 0.3040031
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2630430, 0.2647876

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2377

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 424

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1426624, upper bound: 0.1421633
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1430944, upper bound: 0.1420933
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2656900, 0.2651045
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2294072, 0.2290774
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2219995, 0.2201772
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4683316, 0.4696205
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2504144, 0.2511547
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3394623, 0.3392262
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2941968, 0.2938408
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3290603, 0.3298719
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3040109, 0.3050719
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2644299, 0.2634007

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1839

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1394343, upper bound: 0.1379733
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1380723, upper bound: 0.1393998
time: 3.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1441838, upper bound: 0.1438872
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1442772, upper bound: 0.1437912
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1439576, upper bound: 0.1420913
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1439061, upper bound: 0.1434427
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1437782, upper bound: 0.1436493
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1438324, upper bound: 0.1443059
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1436342, upper bound: 0.1445037
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1429776, upper bound: 0.1444500
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1415937, upper bound: 0.1412566
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1409918, upper bound: 0.1418592
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1409170, upper bound: 0.1427878
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1419415, upper bound: 0.1417621
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1426624, upper bound: 0.1421633
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1430944, upper bound: 0.1420933
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1394343, upper bound: 0.1379733
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.83
Output dim: 1, lower bound: -0.1380723, upper bound: 0.1393998

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2674327, 0.2674625
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2266365, 0.2268785
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2158292, 0.2144182
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4566338, 0.4543328
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2392451, 0.2400825
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3373020, 0.3344007
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2908201, 0.2907470
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3275728, 0.3268193
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2965441, 0.2964945
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2666180, 0.2663700

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1417430, upper bound: 0.1422737
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1425663, upper bound: 0.1414509
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2673618, 0.2675334
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2265056, 0.2270094
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2141988, 0.2160487
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4550335, 0.4559331
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2406148, 0.2387128
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3340161, 0.3376861
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2904530, 0.2911142
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3280833, 0.3263087
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964854, 0.2965529
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2667620, 0.2662259

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 2377

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1409627, upper bound: 0.1403115
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1409627, upper bound: 0.1403115
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2656513, 0.2631435
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2292247, 0.2295975
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2323827, 0.2326887
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4654064, 0.4659319
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2422824, 0.2436178
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3409967, 0.3413694
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2925789, 0.2929189
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3283186, 0.3284876
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2997921, 0.3002547
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2684802, 0.2651989

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 1747

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 424

### Candidate
type: DSZ, layer: 3, pos: 2866

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1406622, upper bound: 0.1387507
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1406622, upper bound: 0.1387507
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2630427, 0.2657521
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2292247, 0.2295977
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2324692, 0.2326021
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4666328, 0.4647057
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2441499, 0.2417500
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3409848, 0.3413813
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2926247, 0.2928733
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3297515, 0.3270546
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.3002458, 0.2998009
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2655910, 0.2680882

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 424
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1433526, upper bound: 0.1422703
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1427336, upper bound: 0.1428891
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2669040, 0.2668666
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2254210, 0.2260292
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2330276, 0.2332319
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4674032, 0.4667799
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2452776, 0.2448242
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3397696, 0.3402870
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2939854, 0.2942179
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3313732, 0.3301182
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2964649, 0.2969441
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2663684, 0.2656038

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2005
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1102
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2215
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1258
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2866
type: DSZ, layer: 3, pos: 2378
type: DSZ, layer: 3, pos: 2377
type: DSZ, layer: 3, pos: 2483
type: DSZ, layer: 3, pos: 180
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1500

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1400576, upper bound: 0.1410026
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.1411323, upper bound: 0.1399107
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.6448908, -6.9110246, -7.6448908, -6.9110246, -0.2667657, 0.2670047
1: 2.5623522, 3.0937791, 2.5623522, 3.0937791, -0.2256563, 0.2257938
2: -4.9287162, -4.3579311, -4.9287162, -4.3579311, -0.2330124, 0.2332472
3: -14.4969625, -13.4901037, -14.4969625, -13.4901037, -0.4674807, 0.4667025
4: -3.0554028, -2.4088497, -3.0554028, -2.4088497, -0.2453564, 0.2447454
5: -8.6014805, -7.7889137, -8.6014805, -7.7889137, -0.3399026, 0.3401539
6: -4.5145478, -3.8793623, -4.5145478, -3.8793623, -0.2939239, 0.2942796
7: -8.2884521, -7.7425776, -8.2884521, -7.7425776, -0.3313820, 0.3301092
8: -1.2437325, -0.5141001, -1.2437325, -0.5141001, -0.2969353, 0.2964737
9: -7.3487859, -6.6409111, -7.3487859, -6.6409111, -0.2659959, 0.2659764

Time for backsubstitution: 22.37 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.72 + 560.02 = 616.75 seconds
