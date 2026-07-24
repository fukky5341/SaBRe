## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.7540085754
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (5.2109342, 8.2797604, 5.2109342, 8.2797604, -3.0688262, 3.0688262)
1: (-21.6256638, -17.3819923, -21.6256638, -17.3819923, -4.2436714, 4.2436714)
2: (-5.6255550, -2.4815617, -5.6255550, -2.4815617, -3.1439934, 3.1439934)
3: (-14.0028372, -10.9323034, -14.0028372, -10.9323034, -3.0705338, 3.0705338)
4: (-9.2312660, -6.2723002, -9.2312660, -6.2723002, -2.9589658, 2.9589658)
5: (-7.6828470, -4.8713017, -7.6828470, -4.8713017, -2.8115454, 2.8115454)
6: (-5.5924163, -2.8401895, -5.5924163, -2.8401895, -2.7522268, 2.7522268)
7: (-11.0651436, -7.1901598, -11.0651436, -7.1901598, -3.8749838, 3.8749838)
8: (-4.1027942, -0.9745383, -4.1027942, -0.9745383, -3.1282558, 3.1282558)
9: (-4.8675470, -1.8201666, -4.8675470, -1.8201666, -3.0473804, 3.0473804)

## BASE Result
execution time: IAR + LP analysis = 13.79 + 34.31 = 48.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.4199218, upper bound: 2.4199192


# Binary Search by BASE starts (time budget: 3551.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.068826198577881
rel_dist={0: [-1.7599642570566427, 1.7599639862673033]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNREACHABLE, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.8390512466430664

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.669461727142334
rel_dist={0: [-1.0059846949496585, 1.0059850175342175]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.754256248474121
rel_dist={0: [-1.1909365767727103, 1.1909392632351246]}

## Binary Search Result
Binary search time: 237.30 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3314.60 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8823438, upper bound: 1.8590148
time: 6.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8826747, upper bound: 1.8826728
time: 9.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.62
Output dim: 0, lower bound: -1.8823438, upper bound: 1.8590148
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.62
Output dim: 0, lower bound: -1.8826747, upper bound: 1.8826728

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 5.2188349, 8.2537851, 5.2113304, 8.2783957, -3.0595608, 3.0424547
1: -21.6156731, -17.4120255, -21.6251812, -17.3835697, -3.6511197, 3.6273913
2: -5.6202016, -2.4923961, -5.6252823, -2.4821320, -3.0937490, 3.0852704
3: -13.9943714, -10.9491730, -14.0024118, -10.9331903, -2.8151197, 2.8071256
4: -9.2273426, -6.2768726, -9.2310562, -6.2725420, -2.6892519, 2.6882565
5: -7.6781301, -4.8753815, -7.6825843, -4.8715191, -2.4904795, 2.4889212
6: -5.5776024, -2.8418746, -5.5916352, -2.8402786, -2.5561924, 2.5698071
7: -11.0627842, -7.1975846, -11.0650225, -7.1905489, -3.8722353, 3.8674378
8: -4.0958204, -0.9894047, -4.1024461, -0.9753199, -2.7622747, 2.7528400
9: -4.8517962, -1.8295510, -4.8667183, -1.8206251, -2.8671951, 2.8777599

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590139
time: 11.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590118
time: 9.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 5.1546121, 8.2800674, 5.2109399, 8.2797527, -3.1251407, 3.0691276
1: -21.6784897, -17.3789444, -21.6256599, -17.3820038, -3.7121401, 3.6604400
2: -5.6456547, -2.4789648, -5.6255507, -2.4815698, -3.1201344, 3.1026750
3: -14.0351133, -10.9311829, -14.0028324, -10.9323130, -2.8640065, 2.8257694
4: -9.2350531, -6.2701120, -9.2312622, -6.2723045, -2.6978312, 2.6974244
5: -7.6861029, -4.8690000, -7.6828451, -4.8713055, -2.5068359, 2.4956679
6: -5.5981522, -2.8262198, -5.5924053, -2.8401902, -2.5766106, 2.5864091
7: -11.0796404, -7.1885257, -11.0651426, -7.1901627, -3.8894777, 3.8766170
8: -4.1339159, -0.9734697, -4.1027918, -0.9745529, -2.7942386, 2.7687097
9: -4.8691473, -1.7886271, -4.8675323, -1.8201692, -2.8847256, 2.9163451

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8823437
time: 58.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590120, upper bound: 1.8826748
time: 7.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 80.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 80.04
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590139
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 80.04
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590118
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 80.04
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8823437
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 80.04
Output dim: 0, lower bound: -1.8590120, upper bound: 1.8826748

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 5.2188349, 8.2537851, 5.2188349, 8.2537851, -3.0349503, 3.0349503
1: -21.6156731, -17.4120255, -21.6156731, -17.4120255, -3.6227612, 3.6227612
2: -5.6202016, -2.4923961, -5.6202016, -2.4923961, -3.0801096, 3.0801091
3: -13.9943714, -10.9491730, -13.9943714, -10.9491730, -2.7971425, 2.7971425
4: -9.2273426, -6.2768726, -9.2273426, -6.2768726, -2.6842914, 2.6842904
5: -7.6781301, -4.8753815, -7.6781301, -4.8753815, -2.4830828, 2.4830828
6: -5.5776024, -2.8418746, -5.5776024, -2.8418746, -2.5547299, 2.5547299
7: -11.0627842, -7.1975846, -11.0627842, -7.1975846, -3.8651996, 3.8651996
8: -4.0958204, -0.9894047, -4.0958204, -0.9894047, -2.7482166, 2.7482171
9: -4.8517962, -1.8295510, -4.8517962, -1.8295510, -2.8628473, 2.8628478

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8590076
time: 7.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8590075
time: 5.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 5.2188349, 8.2537851, 5.1546121, 8.2800674, -3.0612326, 3.0991731
1: -21.6156731, -17.4120255, -21.6784897, -17.3789444, -3.6560564, 3.6822238
2: -5.6202016, -2.4923961, -5.6456547, -2.4789648, -3.0974379, 3.1057401
3: -13.9943714, -10.9491730, -14.0351133, -10.9311829, -2.8152466, 2.8382893
4: -9.2273426, -6.2768726, -9.2350531, -6.2701120, -2.6932602, 2.6925857
5: -7.6781301, -4.8753815, -7.6861029, -4.8690000, -2.4895043, 2.4902105
6: -5.5776024, -2.8418746, -5.5981522, -2.8262198, -2.5705023, 2.5753722
7: -11.0627842, -7.1975846, -11.0796404, -7.1885257, -3.8742585, 3.8820558
8: -4.0958204, -0.9894047, -4.1339159, -0.9734697, -2.7641292, 2.7793775
9: -4.8517962, -1.8295510, -4.8691473, -1.7886271, -2.9006147, 2.8802352

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8590098
time: 7.07 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8590079
time: 9.83 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 5.1546121, 8.2800674, 5.2188349, 8.2537851, -3.0991731, 3.0612326
1: -21.6784897, -17.3789444, -21.6156731, -17.4120255, -3.6822238, 3.6560559
2: -5.6456547, -2.4789648, -5.6202016, -2.4923961, -3.1057396, 3.0974374
3: -14.0351133, -10.9311829, -13.9943714, -10.9491730, -2.8382888, 2.8152466
4: -9.2350531, -6.2701120, -9.2273426, -6.2768726, -2.6925864, 2.6932602
5: -7.6861029, -4.8690000, -7.6781301, -4.8753815, -2.4902105, 2.4895034
6: -5.5981522, -2.8262198, -5.5776024, -2.8418746, -2.5753722, 2.5705023
7: -11.0796404, -7.1885257, -11.0627842, -7.1975846, -3.8820558, 3.8742585
8: -4.1339159, -0.9734697, -4.0958204, -0.9894047, -2.7793775, 2.7641287
9: -4.8691473, -1.7886271, -4.8517962, -1.8295510, -2.8802347, 2.9006152

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8823331
time: 8.61 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8823332
time: 8.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 5.1546121, 8.2800674, 5.1546121, 8.2800674, -3.1254554, 3.1254554
1: -21.6784897, -17.3789444, -21.6784897, -17.3789444, -3.7006259, 3.7006264
2: -5.6456547, -2.4789648, -5.6456547, -2.4789648, -3.1168623, 3.1168623
3: -14.0351133, -10.9311829, -14.0351133, -10.9311829, -2.8653355, 2.8653355
4: -9.2350531, -6.2701120, -9.2350531, -6.2701120, -2.7017899, 2.7017903
5: -7.6861029, -4.8690000, -7.6861029, -4.8690000, -2.5088921, 2.5088925
6: -5.5981522, -2.8262198, -5.5981522, -2.8262198, -2.5821447, 2.5821447
7: -11.0796404, -7.1885257, -11.0796404, -7.1885257, -3.8911147, 3.8911147
8: -4.1339159, -0.9734697, -4.1339159, -0.9734697, -2.7926970, 2.7926979
9: -4.8691473, -1.7886271, -4.8691473, -1.7886271, -2.9154406, 2.9154410

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8826661
time: 6.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8826657
time: 9.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8590076
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8590075
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8590098
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8590079
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8823331
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8823332
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8826661
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.69
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8826657

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 5.2311001, 8.2532749, 5.2216668, 8.2536688, -3.0225687, 3.0316081
1: -21.6145554, -17.4135971, -21.6154156, -17.4123917, -3.6181912, 3.6177959
2: -5.6183791, -2.4958391, -5.6197834, -2.4931941, -3.0775938, 3.0759821
3: -13.9834099, -10.9496822, -13.9918375, -10.9492903, -2.7856412, 2.7937264
4: -9.2260981, -6.2903361, -9.2270575, -6.2799797, -2.6799769, 2.6704431
5: -7.6674032, -4.8764486, -7.6756535, -4.8756266, -2.4697075, 2.4770465
6: -5.5759182, -2.8433306, -5.5772171, -2.8422120, -2.5530887, 2.5529585
7: -11.0622511, -7.2023306, -11.0626621, -7.1986785, -3.8635726, 3.8603315
8: -4.0944662, -1.0005486, -4.0955110, -0.9919767, -2.7442608, 2.7365303
9: -4.8512554, -1.8306327, -4.8516736, -1.8297997, -2.8557315, 2.8552570

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8518457
time: 12.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8590098
time: 22.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 5.2144313, 8.2697449, 5.2188368, 8.2537842, -3.0393529, 3.0509081
1: -21.6204262, -17.4094563, -21.6156693, -17.4120293, -3.6234345, 3.6268940
2: -5.6299767, -2.4892259, -5.6202021, -2.4923983, -3.0910482, 3.0828543
3: -13.9984980, -10.9355783, -13.9943657, -10.9491711, -2.8007665, 2.8106070
4: -9.2457237, -6.2750554, -9.2273426, -6.2768779, -2.7027292, 2.6837001
5: -7.6816807, -4.8613553, -7.6781273, -4.8753800, -2.4839301, 2.4983215
6: -5.5845861, -2.8414202, -5.5776024, -2.8418763, -2.5625434, 2.5554755
7: -11.0700045, -7.1948195, -11.0627842, -7.1975861, -3.8724184, 3.8679647
8: -4.1131725, -0.9879241, -4.0958214, -0.9894078, -2.7652426, 2.7461500
9: -4.8525538, -1.8245349, -4.8517966, -1.8295524, -2.8675976, 2.8605509

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589475, upper bound: 1.8532785
time: 9.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589465, upper bound: 1.8589456
time: 12.48 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 5.2311001, 8.2532749, 5.1574392, 8.2799501, -3.0488501, 3.0958357
1: -21.6145554, -17.4135971, -21.6782074, -17.3793030, -3.6514821, 3.6772623
2: -5.6183791, -2.4958391, -5.6452274, -2.4797621, -3.0949183, 3.1015964
3: -13.9834099, -10.9496822, -14.0325909, -10.9313011, -2.8037443, 2.8348823
4: -9.2260981, -6.2903361, -9.2347622, -6.2732186, -2.6889505, 2.6787341
5: -7.6674032, -4.8764486, -7.6836329, -4.8692455, -2.4761276, 2.4841828
6: -5.5759182, -2.8433306, -5.5977697, -2.8265524, -2.5688624, 2.5735912
7: -11.0622511, -7.2023306, -11.0795155, -7.1896176, -3.8726335, 3.8771849
8: -4.0944662, -1.0005486, -4.1335907, -0.9760437, -2.7601719, 2.7676647
9: -4.8512554, -1.8306327, -4.8690186, -1.7888947, -2.8935008, 2.8726387

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8751736, upper bound: 1.8518460
time: 5.68 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8751736, upper bound: 1.8590080
time: 6.25 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 5.2144313, 8.2697449, 5.1546164, 8.2800684, -3.0656371, 3.1151285
1: -21.6204262, -17.4094563, -21.6784878, -17.3789406, -3.6567287, 3.6855686
2: -5.6299767, -2.4892259, -5.6456542, -2.4789648, -3.1083784, 3.1084838
3: -13.9984980, -10.9355783, -14.0351086, -10.9311838, -2.8188710, 2.8517537
4: -9.2457237, -6.2750554, -9.2350531, -6.2701178, -2.7117004, 2.6919951
5: -7.6816807, -4.8613553, -7.6861010, -4.8690014, -2.4903512, 2.5054483
6: -5.5845861, -2.8414202, -5.5981512, -2.8262191, -2.5783162, 2.5761166
7: -11.0700045, -7.1948195, -11.0796394, -7.1885266, -3.8814778, 3.8848200
8: -4.1131725, -0.9879241, -4.1339159, -0.9734743, -2.7811537, 2.7773345
9: -4.8525538, -1.8245349, -4.8691468, -1.7886291, -2.9053645, 2.8779378

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8822731, upper bound: 1.8532786
time: 8.03 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8822721, upper bound: 1.8589457
time: 9.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 5.1668591, 8.2795506, 5.2216668, 8.2536688, -3.0868096, 3.0578837
1: -21.6772480, -17.3805122, -21.6154156, -17.4123917, -3.6776700, 3.6510754
2: -5.6437979, -2.4824052, -5.6197834, -2.4931941, -3.1031504, 3.0932989
3: -14.0241909, -10.9316950, -13.9918375, -10.9492903, -2.8268256, 2.8118291
4: -9.2337847, -6.2835779, -9.2270575, -6.2799797, -2.6882510, 2.6794305
5: -7.6754131, -4.8700771, -7.6756535, -4.8756266, -2.4769030, 2.4834580
6: -5.5964842, -2.8276706, -5.5772171, -2.8422120, -2.5737290, 2.5687344
7: -11.0790997, -7.1932607, -11.0626621, -7.1986785, -3.8804212, 3.8694015
8: -4.1325035, -0.9846184, -4.0955110, -0.9919767, -2.7740040, 2.7524409
9: -4.8685961, -1.7897987, -4.8516736, -1.8297997, -2.8730998, 2.8930330

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8751733
time: 21.20 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8823334
time: 17.27 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 5.1504478, 8.2960167, 5.2188368, 8.2537842, -3.1033363, 3.0771799
1: -21.6829510, -17.3763618, -21.6156693, -17.4120293, -3.6831031, 3.6602015
2: -5.6553402, -2.4758077, -5.6202021, -2.4923983, -3.1166048, 3.1002488
3: -14.0390110, -10.9175901, -13.9943657, -10.9491711, -2.8417497, 2.8287086
4: -9.2534161, -6.2682815, -9.2273426, -6.2768779, -2.7110205, 2.6927605
5: -7.6898160, -4.8549776, -7.6781273, -4.8753800, -2.4914613, 2.5047374
6: -5.6052065, -2.8257833, -5.5776024, -2.8418763, -2.5832343, 2.5712223
7: -11.0868425, -7.1857381, -11.0627842, -7.1975861, -3.8892565, 3.8770461
8: -4.1511073, -0.9719954, -4.0958214, -0.9894078, -2.7837996, 2.7620573
9: -4.8699007, -1.7839069, -4.8517966, -1.8295524, -2.8849764, 2.8984337

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589475, upper bound: 1.8766065
time: 9.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589465, upper bound: 1.8822712
time: 13.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 5.1668591, 8.2795506, 5.1574392, 8.2799501, -3.1130910, 3.1221113
1: -21.6772480, -17.3805122, -21.6782074, -17.3793030, -3.6960702, 3.6956511
2: -5.6437979, -2.4824052, -5.6452274, -2.4797621, -3.1142712, 3.1127062
3: -14.0241909, -10.9316950, -14.0325909, -10.9313011, -2.8538694, 2.8619266
4: -9.2337847, -6.2835779, -9.2347622, -6.2732186, -2.6974587, 2.6879573
5: -7.6754131, -4.8700771, -7.6836329, -4.8692455, -2.4955678, 2.5028529
6: -5.5964842, -2.8276706, -5.5977697, -2.8265524, -2.5805025, 2.5803666
7: -11.0790997, -7.1932607, -11.0795155, -7.1896176, -3.8894820, 3.8862548
8: -4.1325035, -0.9846184, -4.1335907, -0.9760437, -2.7886658, 2.7809887
9: -4.8685961, -1.7897987, -4.8690186, -1.7888947, -2.9083080, 2.9078526

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8564548, upper bound: 1.8755055
time: 6.83 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8564548, upper bound: 1.8826656
time: 6.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 5.1504478, 8.2960167, 5.1546164, 8.2800684, -3.1296206, 3.1414003
1: -21.6829510, -17.3763618, -21.6784878, -17.3789406, -3.7015066, 3.7047744
2: -5.6553402, -2.4758077, -5.6456542, -2.4789648, -3.1277266, 3.1196704
3: -14.0390110, -10.9175901, -14.0351086, -10.9311838, -2.8687954, 2.8748720
4: -9.2534161, -6.2682815, -9.2350531, -6.2701178, -2.7202263, 2.7012904
5: -7.6898160, -4.8549776, -7.6861010, -4.8690014, -2.5098429, 2.5241261
6: -5.6052065, -2.8257833, -5.5981512, -2.8262191, -2.5900083, 2.5828629
7: -11.0868425, -7.1857381, -11.0796394, -7.1885266, -3.8983159, 3.8939013
8: -4.1511073, -0.9719954, -4.1339159, -0.9734743, -2.7994604, 2.7906270
9: -4.8699007, -1.7839069, -4.8691468, -1.7886291, -2.9201813, 2.9132586

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8635559, upper bound: 1.8769354
time: 12.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8635549, upper bound: 1.8826025
time: 14.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 41.74 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8518457
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8590098
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8589475, upper bound: 1.8532785
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8589465, upper bound: 1.8589456
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8751736, upper bound: 1.8518460
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8751736, upper bound: 1.8590080
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8822731, upper bound: 1.8532786
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8822721, upper bound: 1.8589457
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8751733
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8823334
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8589475, upper bound: 1.8766065
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8589465, upper bound: 1.8822712
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8564548, upper bound: 1.8755055
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8564548, upper bound: 1.8826656
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8635559, upper bound: 1.8769354
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 41.74
Output dim: 0, lower bound: -1.8635549, upper bound: 1.8826025

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 5.2311001, 8.2532749, 5.2311001, 8.2532749, -3.0221748, 3.0221748
1: -21.6145554, -17.4135971, -21.6145554, -17.4135971, -3.6150150, 3.6150150
2: -5.6183791, -2.4958391, -5.6183791, -2.4958391, -3.0747061, 3.0747061
3: -13.9834099, -10.9496822, -13.9834099, -10.9496822, -2.7850218, 2.7850218
4: -9.2260981, -6.2903361, -9.2260981, -6.2903361, -2.6695337, 2.6695335
5: -7.6674032, -4.8764486, -7.6674032, -4.8764486, -2.4673095, 2.4673095
6: -5.5759182, -2.8433306, -5.5759182, -2.8433306, -2.5519562, 2.5519562
7: -11.0622511, -7.2023306, -11.0622511, -7.2023306, -3.8599205, 3.8599205
8: -4.0944662, -1.0005486, -4.0944662, -1.0005486, -2.7355089, 2.7355080
9: -4.8512554, -1.8306327, -4.8512554, -1.8306327, -2.8508978, 2.8508973

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8461171, upper bound: 1.8517858
time: 6.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8517841, upper bound: 1.8517845
time: 14.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 5.2311001, 8.2532749, 5.2144313, 8.2697449, -3.0386448, 3.0388436
1: -21.6145554, -17.4135971, -21.6204262, -17.4094563, -3.6192751, 3.6193089
2: -5.6183791, -2.4958391, -5.6299767, -2.4892259, -3.0819721, 3.0873022
3: -13.9834099, -10.9496822, -13.9984980, -10.9355783, -2.7992907, 2.8007083
4: -9.2260981, -6.2903361, -9.2457237, -6.2750554, -2.6848989, 2.6891584
5: -7.6674032, -4.8764486, -7.6816807, -4.8613553, -2.4823389, 2.4808178
6: -5.5759182, -2.8433306, -5.5845861, -2.8414202, -2.5541949, 2.5610707
7: -11.0622511, -7.2023306, -11.0700045, -7.1948195, -3.8674316, 3.8676739
8: -4.0944662, -1.0005486, -4.1131725, -0.9879241, -2.7477479, 2.7538662
9: -4.8512554, -1.8306327, -4.8525538, -1.8245349, -2.8548870, 2.8522129

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8461171, upper bound: 1.8589478
time: 6.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8517841, upper bound: 1.8589463
time: 15.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 5.2144313, 8.2697449, 5.2258706, 8.2535000, -3.0390687, 3.0438743
1: -21.6204262, -17.4094563, -21.6046085, -17.4136715, -3.6219788, 3.6155066
2: -5.6299767, -2.4892259, -5.6186185, -2.5107465, -3.0725060, 3.0813565
3: -13.9984980, -10.9355783, -13.9925489, -10.9549026, -2.7948666, 2.8089705
4: -9.2457237, -6.2750554, -9.2257242, -6.2775965, -2.7008824, 2.6804671
5: -7.6816807, -4.8613553, -7.6728601, -4.8760853, -2.4830647, 2.4931550
6: -5.5845861, -2.8414202, -5.5635805, -2.8426797, -2.5618854, 2.5414071
7: -11.0700045, -7.1948195, -11.0617990, -7.2103519, -3.8596525, 3.8669796
8: -4.1131725, -0.9879241, -4.0945096, -0.9943023, -2.7601495, 2.7449970
9: -4.8525538, -1.8245349, -4.8472843, -1.8302908, -2.8662610, 2.8561163

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8532786
time: 10.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8532788
time: 6.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 5.2144532, 8.2697420, 5.2145300, 8.2832642, -3.0688109, 3.0552120
1: -21.6203766, -17.4094563, -21.6187687, -17.3549843, -3.6799493, 3.6346831
2: -5.6299748, -2.4892652, -5.7142801, -2.4889770, -3.1057472, 3.1299944
3: -13.9984932, -10.9355831, -14.0308514, -10.9452438, -2.8095560, 2.8482935
4: -9.2457218, -6.2750564, -9.2381086, -6.2737761, -2.7130084, 2.6906495
5: -7.6816759, -4.8613563, -7.6809130, -4.8468103, -2.5133057, 2.5052204
6: -5.5845404, -2.8414230, -5.5850592, -2.7718768, -2.5978818, 2.5752621
7: -11.0700026, -7.1948647, -11.1243448, -7.1921196, -3.8778830, 3.9294801
8: -4.1131692, -0.9879382, -4.1234040, -0.9880693, -2.7672968, 2.7749591
9: -4.8525443, -1.8245373, -4.8550148, -1.8117762, -2.8840094, 2.8663936

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8589463
time: 10.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8589465
time: 16.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 5.2311001, 8.2532749, 5.1668591, 8.2795506, -3.0484505, 3.0864158
1: -21.6145554, -17.4135971, -21.6772480, -17.3805122, -3.6482944, 3.6744943
2: -5.6183791, -2.4958391, -5.6437979, -2.4824052, -3.0920229, 3.1002636
3: -13.9834099, -10.9496822, -14.0241909, -10.9316950, -2.8031249, 2.8262057
4: -9.2260981, -6.2903361, -9.2337847, -6.2835779, -2.6785212, 2.6778073
5: -7.6674032, -4.8764486, -7.6754131, -4.8700771, -2.4737215, 2.4745045
6: -5.5759182, -2.8433306, -5.5964842, -2.8276706, -2.5677323, 2.5725963
7: -11.0622511, -7.2023306, -11.0790997, -7.1932607, -3.8689904, 3.8767691
8: -4.0944662, -1.0005486, -4.1325035, -0.9846184, -2.7514181, 2.7665863
9: -4.8512554, -1.8306327, -4.8685961, -1.7897987, -2.8886733, 2.8682661

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8694447, upper bound: 1.8517855
time: 6.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8751117, upper bound: 1.8517848
time: 27.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 48.58 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8461171, upper bound: 1.8517858
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8517841, upper bound: 1.8517845
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8461171, upper bound: 1.8589478
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8517841, upper bound: 1.8589463
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8532786
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8532788
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8589463
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8532787, upper bound: 1.8589465
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8694447, upper bound: 1.8517855
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 48.58
Output dim: 0, lower bound: -1.8751117, upper bound: 1.8517848
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8751736, upper bound: 1.8590080
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8822731, upper bound: 1.8532786
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8822721, upper bound: 1.8589457
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8751733
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8518460, upper bound: 1.8823334
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8589475, upper bound: 1.8766065
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8589465, upper bound: 1.8822712
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8564548, upper bound: 1.8755055
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8564548, upper bound: 1.8826656
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8635559, upper bound: 1.8769354
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 48.58
Output dim: 0, lower bound: -1.8635549, upper bound: 1.8826025
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.068826198577881
rel_dist={0: [-1.882689891073806, 1.882689645160882]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.9238462448120117
rel_dist={0: [-1.4958789978991227, 1.4958786753824898]}

## Binary search (step 2) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=5, k_high=6, k_mid=5, eps_mid=0.0195312, abs_max=3.008641242980957
rel_dist={0: [-1.6317345307279227, 1.631736453304411]}

## Binary search (step 3) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7590897, upper bound: 1.7393755
time: 8.02 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7599516, upper bound: 1.7599520
time: 9.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.96
Output dim: 0, lower bound: -1.7590897, upper bound: 1.7393755
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.96
Output dim: 0, lower bound: -1.7599516, upper bound: 1.7599520

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 5.2188349, 8.2537851, 5.2120819, 8.2758198, -3.0569849, 3.0417032
1: -21.6156731, -17.4120255, -21.6242523, -17.3865490, -3.4897857, 3.4685588
2: -5.6202016, -2.4923961, -5.6247644, -2.4832058, -3.0047770, 2.9972124
3: -13.9943714, -10.9491730, -14.0016050, -10.9348640, -2.7131281, 2.7059789
4: -9.2273426, -6.2768726, -9.2306662, -6.2729983, -2.5903215, 2.5894461
5: -7.6781301, -4.8753815, -7.6820936, -4.8719254, -2.4095421, 2.4081340
6: -5.5776024, -2.8418746, -5.5901628, -2.8404491, -2.4706078, 2.4827859
7: -11.0627842, -7.1975846, -11.0647964, -7.1912889, -3.8669224, 3.8623924
8: -4.0958204, -0.9894047, -4.1017861, -0.9767926, -2.6785297, 2.6700940
9: -4.8517962, -1.8295510, -4.8651576, -1.8215024, -2.7592359, 2.7686768

Time for backsubstitution: 14.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7393731
time: 9.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7393725
time: 8.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 5.1546121, 8.2800674, 5.2109394, 8.2797518, -3.1251397, 3.0691280
1: -21.6784897, -17.3789444, -21.6256599, -17.3820076, -3.5537696, 3.5011964
2: -5.6456547, -2.4789648, -5.6255531, -2.4815741, -3.0325985, 3.0147738
3: -14.0351133, -10.9311829, -14.0028324, -10.9323139, -2.7636046, 2.7256565
4: -9.2350531, -6.2701120, -9.2312613, -6.2723050, -2.5994353, 2.5989950
5: -7.6861029, -4.8690000, -7.6828442, -4.8713055, -2.4262176, 2.4155054
6: -5.5981522, -2.8262198, -5.5924034, -2.8401916, -2.4906440, 2.5009696
7: -11.0796404, -7.1885257, -11.0651426, -7.1901622, -3.8852110, 3.8714733
8: -4.1339159, -0.9734697, -4.1027913, -0.9745548, -2.7113714, 2.6859360
9: -4.8691473, -1.7886271, -4.8675313, -1.8201714, -2.7770534, 2.8088207

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7590895
time: 6.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7599517
time: 7.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.53 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.53
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7393731
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 28.53
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7393725
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7590895
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7599517

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 5.1546121, 8.2800674, 5.2188349, 8.2537851, -3.0991731, 3.0612326
1: -21.6784897, -17.3789444, -21.6156731, -17.4120255, -3.5238566, 3.4976897
2: -5.6456547, -2.4789648, -5.6202016, -2.4923961, -3.0182037, 3.0099020
3: -14.0351133, -10.9311829, -13.9943714, -10.9491730, -2.7381778, 2.7151346
4: -9.2350531, -6.2701120, -9.2273426, -6.2768726, -2.5941901, 2.5948653
5: -7.6861029, -4.8690000, -7.6781301, -4.8753815, -2.4100485, 2.4093418
6: -5.5981522, -2.8262198, -5.5776024, -2.8418746, -2.4899344, 2.4850645
7: -11.0796404, -7.1885257, -11.0627842, -7.1975846, -3.8778391, 3.8697205
8: -4.1339159, -0.9734697, -4.0958204, -0.9894047, -2.6965108, 2.6818538
9: -4.8691473, -1.7886271, -4.8517962, -1.8295510, -2.7727137, 2.7930937

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7590808
time: 6.09 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7590813
time: 14.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 5.1546121, 8.2800674, 5.1546121, 8.2800674, -3.1254554, 3.1254554
1: -21.6784897, -17.3789444, -21.6784897, -17.3789444, -3.5413833, 3.5413837
2: -5.6456547, -2.4789648, -5.6456547, -2.4789648, -3.0289612, 3.0289617
3: -14.0351133, -10.9311829, -14.0351133, -10.9311829, -2.7649364, 2.7649360
4: -9.2350531, -6.2701120, -9.2350531, -6.2701120, -2.6033621, 2.6033618
5: -7.6861029, -4.8690000, -7.6861029, -4.8690000, -2.4282761, 2.4282765
6: -5.5981522, -2.8262198, -5.5981522, -2.8262198, -2.4961782, 2.4961782
7: -11.0796404, -7.1885257, -11.0796404, -7.1885257, -3.8837032, 3.8837032
8: -4.1339159, -0.9734697, -4.1339159, -0.9734697, -2.7099237, 2.7099247
9: -4.8691473, -1.7886271, -4.8691473, -1.7886271, -2.8077688, 2.8077688

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7599438
time: 7.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7599425
time: 6.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.37 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.37
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7590808
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.37
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7590813
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.37
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7599438
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.37
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7599425

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 5.1668591, 8.2795506, 5.2225933, 8.2536306, -3.0867715, 3.0569572
1: -21.6772480, -17.3805122, -21.6153336, -17.4125137, -3.5187693, 3.4924355
2: -5.6437979, -2.4824052, -5.6196480, -2.4934545, -3.0153322, 3.0056391
3: -14.0241909, -10.9316950, -13.9910107, -10.9493265, -2.7266531, 2.7108631
4: -9.2337847, -6.2835779, -9.2269630, -6.2809939, -2.5888314, 2.5809484
5: -7.6754131, -4.8700771, -7.6748438, -4.8757057, -2.3965063, 2.4023399
6: -5.5964842, -2.8276706, -5.5770893, -2.8423209, -2.4881806, 2.4831979
7: -11.0790997, -7.1932607, -11.0626240, -7.1990404, -3.8754930, 3.8640203
8: -4.1325035, -0.9846184, -4.0954094, -0.9928179, -2.6900940, 2.6700659
9: -4.8685961, -1.7897987, -4.8516316, -1.8298788, -2.7651033, 2.7850823

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7330249, upper bound: 1.7533564
time: 5.89 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7330242, upper bound: 1.7590199
time: 5.86 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 5.1504478, 8.2960167, 5.2188358, 8.2537842, -3.1033363, 3.0771809
1: -21.6829510, -17.3763618, -21.6156731, -17.4120312, -3.5246172, 3.5015507
2: -5.6553402, -2.4758077, -5.6202030, -2.4923985, -3.0290680, 3.0124569
3: -14.0390110, -10.9175901, -13.9943657, -10.9491711, -2.7414451, 2.7285976
4: -9.2534161, -6.2682815, -9.2273426, -6.2768774, -2.6126251, 2.5937505
5: -7.6898160, -4.8549776, -7.6781282, -4.8753805, -2.4112978, 2.4244094
6: -5.6052065, -2.8257833, -5.5776019, -2.8418763, -2.4977965, 2.4857795
7: -11.0868425, -7.1857381, -11.0627861, -7.1975870, -3.8854427, 3.8707857
8: -4.1511073, -0.9719954, -4.0958214, -0.9894094, -2.7009330, 2.6790228
9: -4.8699007, -1.7839069, -4.8517962, -1.8295522, -2.7769723, 2.7909112

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393101, upper bound: 1.7533563
time: 8.68 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393095, upper bound: 1.7590199
time: 8.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 5.1668591, 8.2795506, 5.1583652, 8.2799129, -3.1130538, 3.1211853
1: -21.6772480, -17.3805122, -21.6781120, -17.3794231, -3.5365162, 3.5361366
2: -5.6437979, -2.4824052, -5.6450891, -2.4800224, -3.0260859, 3.0246758
3: -14.0241909, -10.9316950, -14.0317650, -10.9313383, -2.7534094, 2.7606750
4: -9.2337847, -6.2835779, -9.2346668, -6.2742348, -2.5980072, 2.5894380
5: -7.6754131, -4.8700771, -7.6828275, -4.8693275, -2.4147167, 2.4212823
6: -5.5964842, -2.8276706, -5.5976434, -2.8266635, -2.4944243, 2.4943023
7: -11.0790997, -7.1932607, -11.0794764, -7.1899757, -3.8813562, 3.8779964
8: -4.1325035, -0.9846184, -4.1334867, -0.9768858, -2.7050343, 2.6981115
9: -4.8685961, -1.7897987, -4.8689780, -1.7889841, -2.8001604, 2.7997513

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7373006, upper bound: 1.7542197
time: 11.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7372999, upper bound: 1.7598850
time: 7.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 5.1504478, 8.2960167, 5.1546173, 8.2800674, -3.1296196, 3.1413994
1: -21.6829510, -17.3763618, -21.6784859, -17.3789406, -3.5422626, 3.5452471
2: -5.6553402, -2.4758077, -5.6456552, -2.4789639, -3.0398254, 3.0315142
3: -14.0390110, -10.9175901, -14.0351086, -10.9311848, -2.7682014, 2.7723107
4: -9.2534161, -6.2682815, -9.2350531, -6.2701182, -2.6217966, 2.6022463
5: -7.6898160, -4.8549776, -7.6861000, -4.8689995, -2.4292259, 2.4433446
6: -5.6052065, -2.8257833, -5.5981512, -2.8262198, -2.5040426, 2.4968910
7: -11.0868425, -7.1857381, -11.0796413, -7.1885281, -3.8913107, 3.8847675
8: -4.1511073, -0.9719954, -4.1339164, -0.9734747, -2.7160921, 2.7070942
9: -4.8699007, -1.7839069, -4.8691478, -1.7886314, -2.8120284, 2.8055859

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7435572, upper bound: 1.7542193
time: 16.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7435566, upper bound: 1.7598833
time: 16.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 47.22 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7330249, upper bound: 1.7533564
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7330242, upper bound: 1.7590199
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7393101, upper bound: 1.7533563
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7393095, upper bound: 1.7590199
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7373006, upper bound: 1.7542197
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7372999, upper bound: 1.7598850
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7435572, upper bound: 1.7542193
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 47.22
Output dim: 0, lower bound: -1.7435566, upper bound: 1.7598833

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 5.1668797, 8.2795525, 5.2182827, 8.2831116, -3.1093316, 3.0612698
1: -21.6771946, -17.3805141, -21.6184330, -17.3554535, -3.5280628, 3.4986229
2: -5.6437926, -2.4824538, -5.7137203, -2.4900339, -3.0265169, 3.0485888
3: -14.0241814, -10.9316998, -14.0275040, -10.9454012, -2.7347870, 2.7507658
4: -9.2337818, -6.2835789, -9.2377253, -6.2778916, -2.5987940, 2.5878935
5: -7.6754050, -4.8700800, -7.6776266, -4.8471355, -2.4258785, 2.4081616
6: -5.5964313, -2.8276722, -5.5845408, -2.7723191, -2.5207679, 2.4998007
7: -11.0790968, -7.1933126, -11.1241798, -7.1935749, -3.8855219, 3.8962622
8: -4.1325006, -0.9846320, -4.1229911, -0.9914804, -2.6921215, 2.6967235
9: -4.8685818, -1.7897987, -4.8548479, -1.8121078, -2.7815151, 2.7901201

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590206
time: 5.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590209
time: 9.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 5.1504731, 8.2960167, 5.2145295, 8.2832642, -3.1233501, 3.0814872
1: -21.6828938, -17.3763618, -21.6187687, -17.3549824, -3.5339179, 3.5077400
2: -5.6553373, -2.4758530, -5.7142801, -2.4889765, -3.0402546, 3.0557237
3: -14.0390034, -10.9175968, -14.0308504, -10.9452438, -2.7495832, 2.7607694
4: -9.2534132, -6.2682838, -9.2381086, -6.2737780, -2.6225872, 2.6006982
5: -7.6898079, -4.8549795, -7.6809101, -4.8468094, -2.4406719, 2.4302306
6: -5.6051540, -2.8257859, -5.5850592, -2.7718773, -2.5279429, 2.5024009
7: -11.0868397, -7.1857915, -11.1243439, -7.1921191, -3.8947206, 3.9031367
8: -4.1511016, -0.9720106, -4.1234026, -0.9880714, -2.7029610, 2.7057416
9: -4.8698859, -1.7839074, -4.8550138, -1.8117769, -2.7933822, 2.7959499

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7336453, upper bound: 1.7590211
time: 6.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7336453, upper bound: 1.7590205
time: 8.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 5.1668591, 8.2795506, 5.1653881, 8.2796268, -3.1127677, 3.1141624
1: -21.6772480, -17.3805122, -21.6670494, -17.3810768, -3.5350466, 3.5247507
2: -5.6437979, -2.4824052, -5.6435127, -2.4983678, -3.0075455, 3.0231843
3: -14.0241909, -10.9316950, -14.0299664, -10.9370661, -2.7475114, 2.7590446
4: -9.2337847, -6.2835779, -9.2330484, -6.2749567, -2.5961533, 2.5862036
5: -7.6754131, -4.8700771, -7.6775584, -4.8700361, -2.4138460, 2.4161248
6: -5.5964842, -2.8276706, -5.5836253, -2.8274648, -2.4937692, 2.4802439
7: -11.0790997, -7.1932607, -11.0784979, -7.2027426, -3.8686161, 3.8772049
8: -4.1325035, -0.9846184, -4.1321754, -0.9817803, -2.6999378, 2.6969523
9: -4.8685961, -1.7897987, -4.8644676, -1.7897182, -2.7988300, 2.7953196

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7542197
time: 6.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7542194
time: 14.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 5.1668797, 8.2795525, 5.1540556, 8.3093948, -3.1346140, 3.1254969
1: -21.6771946, -17.3805141, -21.6812172, -17.3223724, -3.5600286, 3.5423298
2: -5.6437926, -2.4824538, -5.7391982, -2.4765959, -3.0372763, 3.0612960
3: -14.0241814, -10.9316998, -14.0682764, -10.9274206, -2.7615366, 2.7769566
4: -9.2337818, -6.2835789, -9.2454281, -6.2711325, -2.6079769, 2.5963926
5: -7.6754050, -4.8700800, -7.6856318, -4.8407645, -2.4401412, 2.4271030
6: -5.5964313, -2.8276722, -5.6051598, -2.7566602, -2.5260611, 2.5109925
7: -11.0790968, -7.1933126, -11.1410484, -7.1844969, -3.8923340, 3.9047441
8: -4.1325006, -0.9846320, -4.1610765, -0.9755483, -2.7070837, 2.7112939
9: -4.8685818, -1.7897987, -4.8722000, -1.7712164, -2.8139248, 2.8047957

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7598840
time: 6.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7598841
time: 6.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 5.1504478, 8.2960167, 5.1616411, 8.2797832, -3.1293354, 3.1343756
1: -21.6829510, -17.3763618, -21.6674290, -17.3805962, -3.5407972, 3.5338616
2: -5.6553402, -2.4758077, -5.6440773, -2.4973133, -3.0212841, 3.0300221
3: -14.0390110, -10.9175901, -14.0333071, -10.9369154, -2.7623043, 2.7706916
4: -9.2534161, -6.2682815, -9.2334347, -6.2708411, -2.6199436, 2.5990124
5: -7.6898160, -4.8549776, -7.6808319, -4.8697062, -2.4283566, 2.4381862
6: -5.6052065, -2.8257833, -5.5841336, -2.8270202, -2.5033860, 2.4828367
7: -11.0868425, -7.1857381, -11.0786648, -7.2012930, -3.8785696, 3.8839607
8: -4.1511073, -0.9719954, -4.1326056, -0.9783740, -2.7109795, 2.7059417
9: -4.8699007, -1.7839069, -4.8646364, -1.7893653, -2.8106966, 2.8011532

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542180
time: 22.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542216
time: 8.26 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 5.1504731, 8.2960167, 5.1503091, 8.3095493, -3.1486349, 3.1457076
1: -21.6828938, -17.3763618, -21.6815891, -17.3219032, -3.5658884, 3.5514417
2: -5.6553373, -2.4758530, -5.7397699, -2.4755397, -3.0510159, 3.0684586
3: -14.0390034, -10.9175968, -14.0716085, -10.9272585, -2.7763329, 2.7843883
4: -9.2534132, -6.2682838, -9.2458181, -6.2670135, -2.6317644, 2.6092052
5: -7.6898079, -4.8549795, -7.6889033, -4.8404336, -2.4552312, 2.4491634
6: -5.6051540, -2.8257859, -5.6056728, -2.7562180, -2.5332437, 2.5135891
7: -11.0868397, -7.1857915, -11.1412134, -7.1830487, -3.9022923, 3.9116287
8: -4.1511016, -0.9720106, -4.1615038, -0.9721384, -2.7181196, 2.7203259
9: -4.8698859, -1.7839074, -4.8723702, -1.7708583, -2.8207545, 2.8106308

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598822
time: 21.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598837
time: 11.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 47.58 seconds
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590206
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590209
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7336453, upper bound: 1.7590211
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7336453, upper bound: 1.7590205
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7542197
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7542194
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7598840
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7598841
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542180
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542216
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598822
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 47.58
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598837

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 5.1738830, 8.2792664, 5.2182827, 8.2831116, -3.1013398, 3.0609837
1: -21.6661873, -17.3821602, -21.6184330, -17.3554535, -3.5166712, 3.4940033
2: -5.6422181, -2.5007520, -5.7137203, -2.4900339, -3.0180283, 3.0299964
3: -14.0223894, -10.9374247, -14.0275040, -10.9454012, -2.7318630, 2.7448897
4: -9.2321672, -6.2843008, -9.2377253, -6.2778916, -2.5874338, 2.5860448
5: -7.6701446, -4.8707867, -7.6776266, -4.8471355, -2.4207277, 2.4051461
6: -5.5824661, -2.8284714, -5.5845408, -2.7723191, -2.5067046, 2.4929171
7: -11.0781221, -7.2060285, -11.1241798, -7.1935749, -3.8806162, 3.8834820
8: -4.1311913, -0.9895134, -4.1229911, -0.9914804, -2.6909418, 2.6914725
9: -4.8640857, -1.7905335, -4.8548479, -1.8121078, -2.7770967, 2.7871943

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7527760
time: 5.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590197
time: 5.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 5.1625476, 8.3090324, 5.2182827, 8.2831116, -3.1153092, 3.0907497
1: -21.6803570, -17.3234425, -21.6184330, -17.3554535, -3.5312414, 3.5251436
2: -5.7378902, -2.4789805, -5.7137203, -2.4900339, -3.0498838, 3.0431876
3: -14.0607224, -10.9277802, -14.0275040, -10.9454012, -2.7558584, 2.7459364
4: -9.2445402, -6.2804675, -9.2377253, -6.2778916, -2.6070204, 2.5991430
5: -7.6782184, -4.8415194, -7.6776266, -4.8471355, -2.4102402, 2.4160247
6: -5.6039882, -2.7576644, -5.5845408, -2.7723191, -2.5116634, 2.5066442
7: -11.1406736, -7.1877799, -11.1241798, -7.1935749, -3.8981752, 3.8866777
8: -4.1601009, -0.9832811, -4.1229911, -0.9914804, -2.7035866, 2.6988027
9: -4.8718128, -1.7720361, -4.8548479, -1.8121078, -2.7733631, 2.7933407

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7473369
time: 10.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7536011
time: 10.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 5.1574750, 8.2957335, 5.2145295, 8.2832642, -3.1153545, 3.0812039
1: -21.6718903, -17.3780174, -21.6187687, -17.3549824, -3.5225272, 3.5031252
2: -5.6537571, -2.4941568, -5.7142801, -2.4889765, -3.0317602, 3.0371318
3: -14.0372038, -10.9233160, -14.0308504, -10.9452438, -2.7466507, 2.7548618
4: -9.2517996, -6.2690086, -9.2381086, -6.2737780, -2.6112289, 2.5988488
5: -7.6845484, -4.8556871, -7.6809101, -4.8468094, -2.4355202, 2.4272146
6: -5.5911927, -2.8265862, -5.5850592, -2.7718773, -2.5138800, 2.4955163
7: -11.0858612, -7.1985073, -11.1243439, -7.1921191, -3.8905697, 3.8903575
8: -4.1498022, -0.9768932, -4.1234026, -0.9880714, -2.7017784, 2.7004902
9: -4.8653889, -1.7846408, -4.8550138, -1.8117769, -2.7889638, 2.7930241

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7336451, upper bound: 1.7527257
time: 17.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7527258
time: 8.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 5.1461530, 8.3254986, 5.2145295, 8.2832642, -3.1293144, 3.1109691
1: -21.6860542, -17.3193226, -21.6187687, -17.3549824, -3.5370975, 3.5343018
2: -5.7495279, -2.4723864, -5.7142801, -2.4889765, -3.0588775, 3.0500145
3: -14.0755281, -10.9136686, -14.0308504, -10.9452438, -2.7709303, 2.7636595
4: -9.2641773, -6.2651782, -9.2381086, -6.2737780, -2.6308117, 2.6119447
5: -7.6926165, -4.8263879, -7.6809101, -4.8468094, -2.4250259, 2.4381146
6: -5.6127319, -2.7557817, -5.5850592, -2.7718773, -2.5213304, 2.5092421
7: -11.1484337, -7.1802740, -11.1243439, -7.1921191, -3.9081469, 3.8933802
8: -4.1787348, -0.9706583, -4.1234026, -0.9880714, -2.7144184, 2.7078209
9: -4.8731141, -1.7661366, -4.8550138, -1.8117769, -2.7852364, 2.7991796

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7336451, upper bound: 1.7472856
time: 8.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7472851
time: 9.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 5.1738830, 8.2792664, 5.1653881, 8.2796268, -3.1057439, 3.1138783
1: -21.6661873, -17.3821602, -21.6670494, -17.3810768, -3.5236616, 3.5232806
2: -5.6422181, -2.5007520, -5.6435127, -2.4983678, -3.0060511, 3.0046411
3: -14.0223894, -10.9374247, -14.0299664, -10.9370661, -2.7458806, 2.7531462
4: -9.2321672, -6.2843008, -9.2330484, -6.2749567, -2.5929203, 2.5843496
5: -7.6701446, -4.8707867, -7.6775584, -4.8700361, -2.4086876, 2.4152527
6: -5.5824661, -2.8284714, -5.5836253, -2.8274648, -2.4797053, 2.4795883
7: -11.0781221, -7.2060285, -11.0784979, -7.2027426, -3.8678408, 3.8644676
8: -4.1311913, -0.9895134, -4.1321754, -0.9817803, -2.6987782, 2.6918559
9: -4.8640857, -1.7905335, -4.8644676, -1.7897182, -2.7943983, 2.7939892

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7479800
time: 9.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7542200
time: 7.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 5.1625476, 8.3090324, 5.1653881, 8.2796268, -3.1170793, 3.1333117
1: -21.6803570, -17.3234425, -21.6670494, -17.3810768, -3.5380878, 3.5482924
2: -5.7378902, -2.4789805, -5.6435127, -2.4983678, -3.0437098, 3.0273757
3: -14.0607224, -10.9277802, -14.0299664, -10.9370661, -2.7650070, 2.7658710
4: -9.2445402, -6.2804675, -9.2330484, -6.2749567, -2.6031075, 2.5880418
5: -7.6782184, -4.8415194, -7.6775584, -4.8700361, -2.4175243, 2.4401114
6: -5.6039882, -2.7576644, -5.5836253, -2.8274648, -2.5041838, 2.5120251
7: -11.1406736, -7.1877799, -11.0784979, -7.2027426, -3.8947334, 3.8831034
8: -4.1601009, -0.9832811, -4.1321754, -0.9817803, -2.7114148, 2.6990061
9: -4.8718128, -1.7720361, -4.8644676, -1.7897182, -2.8022776, 2.8090682

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7479795
time: 11.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7542179
time: 9.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 5.1738830, 8.2792664, 5.1540556, 8.3093948, -3.1266212, 3.1252108
1: -21.6661873, -17.3821602, -21.6812172, -17.3223724, -3.5486379, 3.5377069
2: -5.6422181, -2.5007520, -5.7391982, -2.4765959, -3.0287876, 3.0427036
3: -14.0223894, -10.9374247, -14.0682764, -10.9274206, -2.7586122, 2.7710476
4: -9.2321672, -6.2843008, -9.2454281, -6.2711325, -2.5966115, 2.5945444
5: -7.6701446, -4.8707867, -7.6856318, -4.8407645, -2.4348712, 2.4240880
6: -5.5824661, -2.8284714, -5.6051598, -2.7566602, -2.5119987, 2.5041089
7: -11.0781221, -7.2060285, -11.1410484, -7.1844969, -3.8864880, 3.8919640
8: -4.1311913, -0.9895134, -4.1610765, -0.9755483, -2.7059288, 2.7060432
9: -4.8640857, -1.7905335, -4.8722000, -1.7712164, -2.8094816, 2.8018708

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7536436
time: 6.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7598833
time: 6.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 5.1625476, 8.3090324, 5.1540556, 8.3093948, -3.1394672, 3.1472831
1: -21.6803570, -17.3234425, -21.6812172, -17.3223724, -3.5632071, 3.5628607
2: -5.7378902, -2.4789805, -5.7391982, -2.4765959, -3.0636654, 3.0622611
3: -14.0607224, -10.9277802, -14.0682764, -10.9274206, -2.7779474, 2.7839808
4: -9.2445402, -6.2804675, -9.2454281, -6.2711325, -2.6162033, 2.6076422
5: -7.6782184, -4.8415194, -7.6856318, -4.8407645, -2.4284048, 2.4349656
6: -5.6039882, -2.7576644, -5.6051598, -2.7566602, -2.5179095, 2.5178361
7: -11.1406736, -7.1877799, -11.1410484, -7.1844969, -3.9040461, 3.9006634
8: -4.1601009, -0.9832811, -4.1610765, -0.9755483, -2.7187443, 2.7133734
9: -4.8718128, -1.7720361, -4.8722000, -1.7712164, -2.8084202, 2.8080158

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7482022
time: 7.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7544630
time: 7.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 29.38 seconds
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7527760
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590197
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7473369
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7536011
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7336451, upper bound: 1.7527257
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7527258
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7336451, upper bound: 1.7472856
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7472851
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7479800
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7542200
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7479795
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316376, upper bound: 1.7542179
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7536436
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7598833
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7482022
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.38
Output dim: 0, lower bound: -1.7316356, upper bound: 1.7544630
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.38
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542180
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.38
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542216
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.38
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598822
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.38
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598837
Binary search (step 3): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.068826198577881
rel_dist={0: [-1.7599642570566427, 1.7599639862673033]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 1716.32 seconds
