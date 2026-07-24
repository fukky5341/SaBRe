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
execution time: IAR + LP analysis = 14.12 + 34.55 = 48.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.4199218, upper bound: 2.4199192


# Binary Search by BASE starts (time budget: 3551.33 seconds, max iter: 100)

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
Binary search time: 238.75 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3312.58 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8823438, upper bound: 1.8590148
time: 6.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8826747, upper bound: 1.8826728
time: 10.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.82
Output dim: 0, lower bound: -1.8823438, upper bound: 1.8590148
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.82
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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590139
time: 11.92 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590118
time: 9.42 seconds

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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8823437
time: 57.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590120, upper bound: 1.8826748
time: 7.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 79.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 79.69
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590139
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 79.69
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8590118
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 79.69
Output dim: 0, lower bound: -1.8590121, upper bound: 1.8823437
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 79.69
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

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8518456
time: 20.27 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8590076
time: 6.95 seconds

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8590098
time: 7.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8590079
time: 9.79 seconds

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8751731
time: 12.96 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8823329
time: 9.23 seconds

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

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8755038
time: 7.01 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590051, upper bound: 1.8826638
time: 9.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.83 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8518456
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8590076
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8518437, upper bound: 1.8590098
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590049, upper bound: 1.8590079
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8751731
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8823329
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590052, upper bound: 1.8755038
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -1.8590051, upper bound: 1.8826638

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 5.2216668, 8.2536688, 5.2311001, 8.2532749, -3.0316081, 3.0225687
1: -21.6154156, -17.4123917, -21.6145554, -17.4135971, -3.6177959, 3.6181912
2: -5.6197834, -2.4931941, -5.6183791, -2.4958391, -3.0759821, 3.0775933
3: -13.9918375, -10.9492903, -13.9834099, -10.9496822, -2.7937260, 2.7856417
4: -9.2270575, -6.2799797, -9.2260981, -6.2903361, -2.6704435, 2.6799772
5: -7.6756535, -4.8756266, -7.6674032, -4.8764486, -2.4770465, 2.4697080
6: -5.5772171, -2.8422120, -5.5759182, -2.8433306, -2.5529585, 2.5530887
7: -11.0626621, -7.1986785, -11.0622511, -7.2023306, -3.8603315, 3.8635726
8: -4.0955110, -0.9919767, -4.0944662, -1.0005486, -2.7365313, 2.7442608
9: -4.8516736, -1.8297997, -4.8512554, -1.8306327, -2.8552570, 2.8557305

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8517858
time: 5.50 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8517843
time: 21.04 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 5.2188368, 8.2537842, 5.2144313, 8.2697449, -3.0509081, 3.0393529
1: -21.6156693, -17.4120293, -21.6204262, -17.4094563, -3.6268940, 3.6234345
2: -5.6202021, -2.4923983, -5.6299767, -2.4892259, -3.0828543, 3.0910482
3: -13.9943657, -10.9491711, -13.9984980, -10.9355783, -2.8106070, 2.8007660
4: -9.2273426, -6.2768779, -9.2457237, -6.2750554, -2.6836996, 2.7027299
5: -7.6781273, -4.8753800, -7.6816807, -4.8613553, -2.4983211, 2.4839306
6: -5.5776024, -2.8418763, -5.5845861, -2.8414202, -2.5554752, 2.5625434
7: -11.0627842, -7.1975861, -11.0700045, -7.1948195, -3.8679647, 3.8724184
8: -4.0958214, -0.9894078, -4.1131725, -0.9879241, -2.7461505, 2.7652421
9: -4.8517966, -1.8295524, -4.8525538, -1.8245349, -2.8605509, 2.8675981

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8589469
time: 6.80 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8589461
time: 11.59 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8751134, upper bound: 1.8532808
time: 7.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8751124, upper bound: 1.8589458
time: 7.11 seconds

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

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8822731, upper bound: 1.8532786
time: 8.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8822721, upper bound: 1.8589457
time: 9.16 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 5.1574392, 8.2799501, 5.2311001, 8.2532749, -3.0958357, 3.0488501
1: -21.6782074, -17.3793030, -21.6145554, -17.4135971, -3.6772623, 3.6514821
2: -5.6452274, -2.4797621, -5.6183791, -2.4958391, -3.1015959, 3.0949187
3: -14.0325909, -10.9313011, -13.9834099, -10.9496822, -2.8348818, 2.8037438
4: -9.2347622, -6.2732186, -9.2260981, -6.2903361, -2.6787348, 2.6889503
5: -7.6836329, -4.8692455, -7.6674032, -4.8764486, -2.4841838, 2.4761271
6: -5.5977697, -2.8265524, -5.5759182, -2.8433306, -2.5735912, 2.5688624
7: -11.0795155, -7.1896176, -11.0622511, -7.2023306, -3.8771849, 3.8726335
8: -4.1335907, -0.9760437, -4.0944662, -1.0005486, -2.7676649, 2.7601724
9: -4.8690186, -1.7888947, -4.8512554, -1.8306327, -2.8726387, 2.8935003

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8751137
time: 5.46 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8751123
time: 8.79 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 5.1546164, 8.2800684, 5.2144313, 8.2697449, -3.1151285, 3.0656371
1: -21.6784878, -17.3789406, -21.6204262, -17.4094563, -3.6855688, 3.6567287
2: -5.6456542, -2.4789648, -5.6299767, -2.4892259, -3.1084833, 3.1083779
3: -14.0351086, -10.9311838, -13.9984980, -10.9355783, -2.8517532, 2.8188710
4: -9.2350531, -6.2701178, -9.2457237, -6.2750554, -2.6919947, 2.7116997
5: -7.6861010, -4.8690014, -7.6816807, -4.8613553, -2.5054479, 2.4903507
6: -5.5981512, -2.8262191, -5.5845861, -2.8414202, -2.5761166, 2.5783165
7: -11.0796394, -7.1885266, -11.0700045, -7.1948195, -3.8848200, 3.8814778
8: -4.1339159, -0.9734743, -4.1131725, -0.9879241, -2.7773342, 2.7811532
9: -4.8691468, -1.7886291, -4.8525538, -1.8245349, -2.8779383, 2.9053655

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8822727
time: 8.22 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8822719
time: 7.84 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 5.1574392, 8.2799501, 5.1668591, 8.2795506, -3.1221113, 3.1130910
1: -21.6782074, -17.3793030, -21.6772480, -17.3805122, -3.6956511, 3.6960702
2: -5.6452274, -2.4797621, -5.6437979, -2.4824052, -3.1127062, 3.1142712
3: -14.0325909, -10.9313011, -14.0241909, -10.9316950, -2.8619270, 2.8538694
4: -9.2347622, -6.2732186, -9.2337847, -6.2835779, -2.6879568, 2.6974587
5: -7.6836329, -4.8692455, -7.6754131, -4.8700771, -2.5028524, 2.4955678
6: -5.5977697, -2.8265524, -5.5964842, -2.8276706, -2.5803666, 2.5805025
7: -11.0795155, -7.1896176, -11.0790997, -7.1932607, -3.8862548, 3.8894820
8: -4.1335907, -0.9760437, -4.1325035, -0.9846184, -2.7809887, 2.7886658
9: -4.8690186, -1.7888947, -4.8685961, -1.7897987, -2.9078522, 2.9083076

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8578878, upper bound: 1.8754451
time: 13.12 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8635545, upper bound: 1.8754446
time: 7.64 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 5.1546164, 8.2800684, 5.1504478, 8.2960167, -3.1414003, 3.1296206
1: -21.6784878, -17.3789406, -21.6829510, -17.3763618, -3.7047749, 3.7015061
2: -5.6456542, -2.4789648, -5.6553402, -2.4758077, -3.1196709, 3.1277270
3: -14.0351086, -10.9311838, -14.0390110, -10.9175901, -2.8748722, 2.8687959
4: -9.2350531, -6.2701178, -9.2534161, -6.2682815, -2.7012901, 2.7202258
5: -7.6861010, -4.8690014, -7.6898160, -4.8549776, -2.5241270, 2.5098424
6: -5.5981512, -2.8262191, -5.6052065, -2.8257833, -2.5828629, 2.5900083
7: -11.0796394, -7.1885266, -11.0868425, -7.1857381, -3.8939013, 3.8983159
8: -4.1339159, -0.9734743, -4.1511073, -0.9719954, -2.7906265, 2.7994602
9: -4.8691468, -1.7886291, -4.8699007, -1.7839069, -2.9132586, 2.9201818

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8578878, upper bound: 1.8826072
time: 29.12 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8635545, upper bound: 1.8826039
time: 23.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 66.96 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8517858
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8517843
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8589469
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8589461
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8751134, upper bound: 1.8532808
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8751124, upper bound: 1.8589458
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8822731, upper bound: 1.8532786
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8822721, upper bound: 1.8589457
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8751137
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8751123
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8822727
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8589461, upper bound: 1.8822719
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8578878, upper bound: 1.8754451
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8635545, upper bound: 1.8754446
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8578878, upper bound: 1.8826072
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 66.96
Output dim: 0, lower bound: -1.8635545, upper bound: 1.8826039

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 5.2286987, 8.2533836, 5.2311001, 8.2532749, -3.0245762, 3.0222836
1: -21.6043549, -17.4140320, -21.6145554, -17.4135971, -3.6064100, 3.6167340
2: -5.6181974, -2.5115390, -5.6183791, -2.4958391, -3.0744829, 3.0590506
3: -13.9900208, -10.9550180, -13.9834099, -10.9496822, -2.7920904, 2.7797418
4: -9.2254410, -6.2806969, -9.2260981, -6.2903361, -2.6672106, 2.6781290
5: -7.6703854, -4.8763285, -7.6674032, -4.8764486, -2.4718795, 2.4688411
6: -5.5631933, -2.8430157, -5.5759182, -2.8433306, -2.5388904, 2.5524313
7: -11.0616789, -7.2114468, -11.0622511, -7.2023306, -3.8593483, 3.8508043
8: -4.0941982, -0.9968705, -4.0944662, -1.0005486, -2.7353706, 2.7391677
9: -4.8471603, -1.8305392, -4.8512554, -1.8306327, -2.8508224, 2.8543949

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8461171
time: 5.47 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8517847
time: 5.19 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 5.2173567, 8.2831478, 5.2311182, 8.2532730, -3.0359163, 3.0520296
1: -21.6185131, -17.3553391, -21.6145077, -17.4136009, -3.6255851, 3.6748595
2: -5.7138572, -2.4897718, -5.6183734, -2.4958801, -3.1227436, 3.0922909
3: -14.0283308, -10.9453659, -13.9834032, -10.9496861, -2.8336320, 2.7944260
4: -9.2378235, -6.2768774, -9.2260942, -6.2903371, -2.6773911, 2.6902533
5: -7.6784363, -4.8470554, -7.6673970, -4.8764496, -2.4839458, 2.4990840
6: -5.5846696, -2.7722111, -5.5758734, -2.8433332, -2.5727301, 2.5908654
7: -11.1242199, -7.1932154, -11.0622492, -7.2023745, -3.9218454, 3.8690338
8: -4.1230931, -0.9906392, -4.0944624, -1.0005596, -2.7653337, 2.7463155
9: -4.8548870, -1.8120260, -4.8512425, -1.8306360, -2.8610978, 2.8721452

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8517841, upper bound: 1.8517845
time: 14.65 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8517841, upper bound: 1.8517846
time: 10.82 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 5.2258706, 8.2535000, 5.2144313, 8.2697449, -3.0438743, 3.0390687
1: -21.6046085, -17.4136715, -21.6204262, -17.4094563, -3.6155071, 3.6219783
2: -5.6186185, -2.5107465, -5.6299767, -2.4892259, -3.0813570, 3.0725060
3: -13.9925489, -10.9549026, -13.9984980, -10.9355783, -2.8089705, 2.7948666
4: -9.2257242, -6.2775965, -9.2457237, -6.2750554, -2.6804676, 2.7008822
5: -7.6728601, -4.8760853, -7.6816807, -4.8613553, -2.4931550, 2.4830647
6: -5.5635805, -2.8426797, -5.5845861, -2.8414202, -2.5414071, 2.5618854
7: -11.0617990, -7.2103519, -11.0700045, -7.1948195, -3.8669796, 3.8596525
8: -4.0945096, -0.9943023, -4.1131725, -0.9879241, -2.7449965, 2.7601490
9: -4.8472843, -1.8302908, -4.8525538, -1.8245349, -2.8561163, 2.8662605

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8532786
time: 6.49 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8532790, upper bound: 1.8589464
time: 5.96 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 5.2145300, 8.2832642, 5.2144532, 8.2697420, -3.0552120, 3.0688109
1: -21.6187687, -17.3549843, -21.6203766, -17.4094563, -3.6346836, 3.6799493
2: -5.7142801, -2.4889770, -5.6299748, -2.4892652, -3.1299944, 3.1057477
3: -14.0308514, -10.9452438, -13.9984932, -10.9355831, -2.8482933, 2.8095555
4: -9.2381086, -6.2737761, -9.2457218, -6.2750564, -2.6906490, 2.7130084
5: -7.6809130, -4.8468103, -7.6816759, -4.8613563, -2.5052204, 2.5133057
6: -5.5850592, -2.7718768, -5.5845404, -2.8414230, -2.5752621, 2.5978816
7: -11.1243448, -7.1921196, -11.0700026, -7.1948647, -3.9294801, 3.8778830
8: -4.1234040, -0.9880693, -4.1131692, -0.9879382, -2.7749591, 2.7672973
9: -4.8550148, -1.8117762, -4.8525443, -1.8245373, -2.8663931, 2.8840094

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589467, upper bound: 1.8532805
time: 10.85 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589467, upper bound: 1.8589462
time: 8.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 5.2311001, 8.2532749, 5.1644650, 8.2796659, -3.0485659, 3.0888100
1: -21.6145554, -17.4135971, -21.6671448, -17.3809547, -3.6500154, 3.6658769
2: -5.6183791, -2.4958391, -5.6436524, -2.4981072, -3.0763769, 3.1001053
3: -13.9834099, -10.9496822, -14.0307875, -10.9370308, -2.7978468, 2.8332520
4: -9.2260981, -6.2903361, -9.2331448, -6.2739439, -2.6870980, 2.6754999
5: -7.6674032, -4.8764486, -7.6783667, -4.8699541, -2.4752569, 2.4790244
6: -5.5759182, -2.8433306, -5.5837507, -2.8273547, -2.5682073, 2.5595367
7: -11.0622511, -7.2023306, -11.0785389, -7.2023869, -3.8598642, 3.8762083
8: -4.0944662, -1.0005486, -4.1322818, -0.9809420, -2.7550774, 2.7664845
9: -4.8512554, -1.8306327, -4.8645101, -1.7896290, -2.8921700, 2.8682070

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8694447, upper bound: 1.8532786
time: 12.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8694447, upper bound: 1.8532785
time: 13.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 5.2311182, 8.2532730, 5.1531291, 8.3094330, -3.0783148, 3.1001439
1: -21.6145077, -17.4136009, -21.6813049, -17.3222599, -3.7062063, 3.6850557
2: -5.6183734, -2.4958801, -5.7393370, -2.4763360, -3.1096201, 3.1339035
3: -13.9834032, -10.9496861, -14.0690985, -10.9273787, -2.8125229, 2.8581326
4: -9.2260942, -6.2903371, -9.2455254, -6.2701135, -2.6992340, 2.6856916
5: -7.6673970, -4.8764496, -7.6864386, -4.8406825, -2.5054955, 2.4911165
6: -5.5758734, -2.8433332, -5.6052871, -2.7565513, -2.5949035, 2.5934505
7: -11.0622492, -7.2023745, -11.1410894, -7.1841388, -3.8781104, 3.9387150
8: -4.0944624, -1.0005596, -4.1611824, -0.9747059, -2.7622237, 2.7791388
9: -4.8512425, -1.8306360, -4.8722448, -1.7711253, -2.9062147, 2.8784871

Time for backsubstitution: 14.50 seconds
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
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393756, upper bound: 1.7590894
time: 5.84 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7599502, upper bound: 1.7599511
time: 12.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.58 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 18.58
Output dim: 0, lower bound: -1.7393756, upper bound: 1.7590894
IS_B2, status: Status.UNKNOWN, split count: 1, time: 18.58
Output dim: 0, lower bound: -1.7599502, upper bound: 1.7599511

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: 5.2120819, 8.2758198, 5.2188349, 8.2537851, -3.0417032, 3.0569849
1: -21.6242523, -17.3865490, -21.6156731, -17.4120255, -3.4685588, 3.4897852
2: -5.6247644, -2.4832058, -5.6202016, -2.4923961, -2.9972134, 3.0047760
3: -14.0016050, -10.9348640, -13.9943714, -10.9491730, -2.7059789, 2.7131276
4: -9.2306662, -6.2729983, -9.2273426, -6.2768726, -2.5894456, 2.5903211
5: -7.6820936, -4.8719254, -7.6781301, -4.8753815, -2.4081335, 2.4095411
6: -5.5901628, -2.8404491, -5.5776024, -2.8418746, -2.4827862, 2.4706078
7: -11.0647964, -7.1912889, -11.0627842, -7.1975846, -3.8623924, 3.8669224
8: -4.1017861, -0.9767926, -4.0958204, -0.9894047, -2.6700935, 2.6785293
9: -4.8651576, -1.8215024, -4.8517962, -1.8295510, -2.7686768, 2.7592363

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7393728
time: 6.44 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7590895
time: 6.25 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: 5.2109394, 8.2797518, 5.1546121, 8.2800674, -3.0691280, 3.1251397
1: -21.6256599, -17.3820076, -21.6784897, -17.3789444, -3.5011964, 3.5537691
2: -5.6255531, -2.4815741, -5.6456547, -2.4789648, -3.0147734, 3.0325980
3: -14.0028324, -10.9323139, -14.0351133, -10.9311829, -2.7256565, 2.7636042
4: -9.2312613, -6.2723050, -9.2350531, -6.2701120, -2.5989943, 2.5994353
5: -7.6828442, -4.8713055, -7.6861029, -4.8690000, -2.4155059, 2.4262185
6: -5.5924034, -2.8401916, -5.5981522, -2.8262198, -2.5009699, 2.4906435
7: -11.0651426, -7.1901622, -11.0796404, -7.1885257, -3.8714743, 3.8852110
8: -4.1027913, -0.9745548, -4.1339159, -0.9734697, -2.6859360, 2.7113714
9: -4.8675313, -1.8201714, -4.8691473, -1.7886271, -2.8088198, 2.7770534

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7590897, upper bound: 1.7393729
time: 7.95 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7590897, upper bound: 1.7599513
time: 6.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.40 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 28.40
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7393728
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 28.40
Output dim: 0, lower bound: -1.7393730, upper bound: 1.7590895
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 28.40
Output dim: 0, lower bound: -1.7590897, upper bound: 1.7393729
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 28.40
Output dim: 0, lower bound: -1.7590897, upper bound: 1.7599513

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393699, upper bound: 1.7527872
time: 10.30 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393699, upper bound: 1.7590813
time: 10.17 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: 5.2188349, 8.2537851, 5.1546121, 8.2800674, -3.0612326, 3.0991731
1: -21.6156731, -17.4120255, -21.6784897, -17.3789444, -3.4976892, 3.5238571
2: -5.6202016, -2.4923961, -5.6456547, -2.4789648, -3.0099020, 3.0182042
3: -13.9943714, -10.9491730, -14.0351133, -10.9311829, -2.7151346, 2.7381778
4: -9.2273426, -6.2768726, -9.2350531, -6.2701120, -2.5948658, 2.5941906
5: -7.6781301, -4.8753815, -7.6861029, -4.8690000, -2.4093423, 2.4100485
6: -5.5776024, -2.8418746, -5.5981522, -2.8262198, -2.4850645, 2.4899344
7: -11.0627842, -7.1975846, -11.0796404, -7.1885257, -3.8697205, 3.8778391
8: -4.0958204, -0.9894047, -4.1339159, -0.9734697, -2.6818538, 2.6965103
9: -4.8517962, -1.8295510, -4.8691473, -1.7886271, -2.7930937, 2.7727137

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7393667
time: 10.95 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7393673
time: 16.66 seconds

## BFS IS instance: IS_B2_A2

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

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7393668
time: 10.38 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7599440
time: 5.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.06 seconds
IS_B1_A2_B1, status: Status.VERIFIED, split count: 3, time: 30.06
Output dim: 0, lower bound: -1.7393699, upper bound: 1.7527872
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 0, lower bound: -1.7393699, upper bound: 1.7590813
IS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 30.06
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7393667
IS_B2_A1_A2, status: Status.VERIFIED, split count: 3, time: 30.06
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7393673
IS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 30.06
Output dim: 0, lower bound: -1.7330820, upper bound: 1.7393668
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 0, lower bound: -1.7393669, upper bound: 1.7599440

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: 5.1546173, 8.2800674, 5.2144313, 8.2697449, -3.1151276, 3.0656362
1: -21.6784859, -17.3789406, -21.6204262, -17.4094563, -3.5255260, 3.4983630
2: -5.6456552, -2.4789639, -5.6299767, -2.4892259, -3.0206919, 3.0208416
3: -14.0351086, -10.9311848, -13.9984980, -10.9355783, -2.7502189, 2.7185650
4: -9.2350531, -6.2701182, -9.2457237, -6.2750554, -2.5929861, 2.6133044
5: -7.6861000, -4.8689995, -7.6816807, -4.8613553, -2.4251208, 2.4101887
6: -5.5981512, -2.8262198, -5.5845861, -2.8414202, -2.4906735, 2.4928789
7: -11.0796413, -7.1885281, -11.0700045, -7.1948195, -3.8788967, 3.8773336
8: -4.1339164, -0.9734747, -4.1131725, -0.9879241, -2.6937056, 2.6988778
9: -4.8691478, -1.7886314, -4.8525538, -1.8245349, -2.7704153, 2.7973619

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7590215
time: 7.71 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393090, upper bound: 1.7590206
time: 6.97 seconds

## BFS IS instance: IS_B2_A2_A2

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

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7435572, upper bound: 1.7542193
time: 15.39 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7435566, upper bound: 1.7598833
time: 16.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 46.36 seconds
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 46.36
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7590215
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 46.36
Output dim: 0, lower bound: -1.7393090, upper bound: 1.7590206
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 46.36
Output dim: 0, lower bound: -1.7435572, upper bound: 1.7542193
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 46.36
Output dim: 0, lower bound: -1.7435566, upper bound: 1.7598833

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 5.1616411, 8.2797832, 5.2144313, 8.2697449, -3.1081038, 3.0653520
1: -21.6674290, -17.3805962, -21.6204262, -17.4094563, -3.5141335, 3.4968958
2: -5.6440773, -2.4973133, -5.6299767, -2.4892259, -3.0192003, 3.0023007
3: -14.0333071, -10.9369154, -13.9984980, -10.9355783, -2.7485995, 2.7126675
4: -9.2334347, -6.2708411, -9.2457237, -6.2750554, -2.5897512, 2.6114516
5: -7.6808319, -4.8697062, -7.6816807, -4.8613553, -2.4199615, 2.4093194
6: -5.5841336, -2.8270202, -5.5845861, -2.8414202, -2.4766188, 2.4922230
7: -11.0786648, -7.2012930, -11.0700045, -7.1948195, -3.8780899, 3.8645926
8: -4.1326056, -0.9783740, -4.1131725, -0.9879241, -2.6925306, 2.6937823
9: -4.8646364, -1.7893653, -4.8525538, -1.8245349, -2.7659826, 2.7960305

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7533557
time: 9.80 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7590203
time: 9.52 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 5.1503091, 8.3095493, 5.2144547, 8.2697420, -3.1194329, 3.0950947
1: -21.6815891, -17.3219032, -21.6203690, -17.4094582, -3.5318861, 3.5480273
2: -5.7397699, -2.4755397, -5.6299725, -2.4892683, -3.0517473, 3.0320315
3: -14.0716085, -10.9272585, -13.9984903, -10.9355860, -2.7622957, 2.7266970
4: -9.2458181, -6.2670135, -9.2457228, -6.2750578, -2.5999441, 2.6232729
5: -7.6889033, -4.8404336, -7.6816730, -4.8613567, -2.4309769, 2.4395561
6: -5.6056728, -2.7562180, -5.5845323, -2.8414218, -2.5073714, 2.5133390
7: -11.1412134, -7.1830487, -11.0700016, -7.1948724, -3.9028149, 3.8869529
8: -4.1615038, -0.9721384, -4.1131673, -0.9879377, -2.7051687, 2.7009311
9: -4.8723702, -1.7708583, -4.8525429, -1.8245392, -2.7754617, 2.8035612

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7393097, upper bound: 1.7533563
time: 7.87 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7393097, upper bound: 1.7590202
time: 11.05 seconds

## BFS IS instance: IS_B2_A2_A2_B1

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

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542180
time: 21.46 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542216
time: 8.15 seconds

## BFS IS instance: IS_B2_A2_A2_B2

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

Time for backsubstitution: 14.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598822
time: 21.11 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598837
time: 11.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 46.99 seconds
IS_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7533557
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7336456, upper bound: 1.7590203
IS_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7393097, upper bound: 1.7533563
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7393097, upper bound: 1.7590202
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542180
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7542216
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598822
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 46.99
Output dim: 0, lower bound: -1.7378925, upper bound: 1.7598837

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 5.1616411, 8.2797832, 5.2101369, 8.2992249, -3.1149564, 3.0696464
1: -21.6674290, -17.3805962, -21.6235313, -17.3524113, -3.5234132, 3.4999342
2: -5.6440773, -2.4973133, -5.7241287, -2.4858088, -3.0233917, 3.0398793
3: -14.0333071, -10.9369154, -14.0349998, -10.9316540, -2.7556047, 2.7526388
4: -9.2334347, -6.2708411, -9.2564859, -6.2719526, -2.5915794, 2.6183939
5: -7.6808319, -4.8697062, -7.6844549, -4.8327599, -2.4384170, 2.4130001
6: -5.5841336, -2.8270202, -5.5920458, -2.7714200, -2.5093455, 2.5026011
7: -11.0786648, -7.2012930, -11.1315832, -7.1893687, -3.8839588, 3.8907881
8: -4.1326056, -0.9783740, -4.1408014, -0.9865868, -2.6945629, 2.7075274
9: -4.8646364, -1.7893653, -4.8557606, -1.8067613, -2.7824273, 2.7994742

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590209
time: 10.08 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7527263
time: 6.28 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 5.1503091, 8.3095493, 5.2101369, 8.2992249, -3.1289258, 3.0994124
1: -21.6815891, -17.3219032, -21.6235313, -17.3524113, -3.5379815, 3.5310740
2: -5.7397699, -2.4755397, -5.7241287, -2.4858088, -3.0560246, 3.0584545
3: -14.0716085, -10.9272585, -14.0349998, -10.9316540, -2.7693043, 2.7536950
4: -9.2458181, -6.2670135, -9.2564859, -6.2719526, -2.6111832, 2.6314878
5: -7.6889033, -4.8404336, -7.6844549, -4.8327599, -2.4388685, 2.4238796
6: -5.6056728, -2.7562180, -5.5920458, -2.7714200, -2.5142117, 2.5163262
7: -11.1412134, -7.1830487, -11.1315832, -7.1893687, -3.9014902, 3.9000378
8: -4.1615038, -0.9721384, -4.1408014, -0.9865868, -2.7072496, 2.7148585
9: -4.8723702, -1.7708583, -4.8557606, -1.8067613, -2.7786927, 2.8056192

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7330242, upper bound: 1.7535994
time: 5.55 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7330242, upper bound: 1.7472857
time: 7.05 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 5.1574750, 8.2957335, 5.1616411, 8.2797832, -3.1223083, 3.1340923
1: -21.6718903, -17.3780174, -21.6674290, -17.3805962, -3.5294104, 3.5323977
2: -5.6537571, -2.4941568, -5.6440773, -2.4973133, -3.0197859, 3.0114813
3: -14.0372038, -10.9233160, -14.0333071, -10.9369154, -2.7606659, 2.7647817
4: -9.2517996, -6.2690086, -9.2334347, -6.2708411, -2.6167107, 2.5971584
5: -7.6845484, -4.8556871, -7.6808319, -4.8697062, -2.4231982, 2.4373145
6: -5.5911927, -2.8265862, -5.5841336, -2.8270202, -2.4893322, 2.4821796
7: -11.0858612, -7.1985073, -11.0786648, -7.2012930, -3.8777895, 3.8712187
8: -4.1498022, -0.9768932, -4.1326056, -0.9783740, -2.7097921, 2.7008462
9: -4.8653889, -1.7846408, -4.8646364, -1.7893653, -2.8062649, 2.7998233

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378943, upper bound: 1.7479282
time: 12.25 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378947, upper bound: 1.7479283
time: 12.29 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 5.1461530, 8.3254986, 5.1616411, 8.2797832, -3.1336303, 3.1402311
1: -21.6860542, -17.3193226, -21.6674290, -17.3805962, -3.5438385, 3.5553837
2: -5.7495279, -2.4723864, -5.6440773, -2.4973133, -3.0527058, 3.0342193
3: -14.0755281, -10.9136686, -14.0333071, -10.9369154, -2.7800760, 2.7776904
4: -9.2641773, -6.2651782, -9.2334347, -6.2708411, -2.6268950, 2.6008444
5: -7.6926165, -4.8263879, -7.6808319, -4.8697062, -2.4320302, 2.4505737
6: -5.6127319, -2.7557817, -5.5841336, -2.8270202, -2.5138512, 2.5146308
7: -11.1484337, -7.1802740, -11.0786648, -7.2012930, -3.8992939, 3.8898392
8: -4.1787348, -0.9706583, -4.1326056, -0.9783740, -2.7222462, 2.7079978
9: -4.8731141, -1.7661366, -4.8646364, -1.7893653, -2.8141489, 2.8158019

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378943, upper bound: 1.7479303
time: 8.79 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378947, upper bound: 1.7479281
time: 7.44 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 5.1574750, 8.2957335, 5.1503091, 8.3095493, -3.1406393, 3.1454244
1: -21.6718903, -17.3780174, -21.6815891, -17.3219032, -3.5544977, 3.5468235
2: -5.6537571, -2.4941568, -5.7397699, -2.4755397, -3.0425224, 3.0498667
3: -14.0372038, -10.9233160, -14.0716085, -10.9272585, -2.7734013, 2.7784808
4: -9.2517996, -6.2690086, -9.2458181, -6.2670135, -2.6204033, 2.6073556
5: -7.6845484, -4.8556871, -7.6889033, -4.8404336, -2.4499612, 2.4461465
6: -5.5911927, -2.8265862, -5.6056728, -2.7562180, -2.5191808, 2.5067041
7: -11.0858612, -7.1985073, -11.1412134, -7.1830487, -3.8964424, 3.8988495
8: -4.1498022, -0.9768932, -4.1615038, -0.9721384, -2.7169371, 2.7150733
9: -4.8653889, -1.7846408, -4.8723702, -1.7708583, -2.8163104, 2.8077054

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378923, upper bound: 1.7535907
time: 10.32 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378927, upper bound: 1.7535906
time: 19.72 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 5.1461530, 8.3254986, 5.1503091, 8.3095493, -3.1530762, 3.1542010
1: -21.6860542, -17.3193226, -21.6815891, -17.3219032, -3.5690670, 3.5699520
2: -5.7495279, -2.4723864, -5.7397699, -2.4755397, -3.0755835, 3.0691085
3: -14.0755281, -10.9136686, -14.0716085, -10.9272585, -2.7930212, 2.7913895
4: -9.2641773, -6.2651782, -9.2458181, -6.2670135, -2.6399879, 2.6204515
5: -7.6926165, -4.8263879, -7.6889033, -4.8404336, -2.4429111, 2.4570465
6: -5.6127319, -2.7557817, -5.6056728, -2.7562180, -2.5275769, 2.5204303
7: -11.1484337, -7.1802740, -11.1412134, -7.1830487, -3.9140186, 3.9073715
8: -4.1787348, -0.9706583, -4.1615038, -0.9721384, -2.7295771, 2.7224042
9: -4.8731141, -1.7661366, -4.8723702, -1.7708583, -2.8202925, 2.8138614

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378923, upper bound: 1.7481525
time: 7.09 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7378927, upper bound: 1.7481527
time: 6.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 27.88 seconds
IS_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7590209
IS_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7273597, upper bound: 1.7527263
IS_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7330242, upper bound: 1.7535994
IS_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7330242, upper bound: 1.7472857
IS_B2_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378943, upper bound: 1.7479282
IS_B2_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378947, upper bound: 1.7479283
IS_B2_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378943, upper bound: 1.7479303
IS_B2_A2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378947, upper bound: 1.7479281
IS_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378923, upper bound: 1.7535907
IS_B2_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378927, upper bound: 1.7535906
IS_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378923, upper bound: 1.7481525
IS_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 27.88
Output dim: 0, lower bound: -1.7378927, upper bound: 1.7481527

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 5.1738830, 8.2792664, 5.2101369, 8.2992249, -3.1025929, 3.0691295
1: -21.6661873, -17.3821602, -21.6235313, -17.3524113, -3.5183425, 3.4957914
2: -5.6422181, -2.5007520, -5.7241287, -2.4858088, -3.0226812, 3.0360847
3: -14.0223894, -10.9374247, -14.0349998, -10.9316540, -2.7443466, 2.7525444
4: -9.2321672, -6.2843008, -9.2564859, -6.2719526, -2.5933728, 2.6048393
5: -7.6701446, -4.8707867, -7.6844549, -4.8327599, -2.4248753, 2.4098763
6: -5.5824661, -2.8284714, -5.5920458, -2.7714200, -2.5076017, 2.5011313
7: -11.0781221, -7.2060285, -11.1315832, -7.1893687, -3.8852186, 3.8853221
8: -4.1311913, -0.9895134, -4.1408014, -0.9865868, -2.6919861, 2.6961386
9: -4.8640857, -1.7905335, -4.8557606, -1.8067613, -2.7767472, 2.7845712

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5844

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7257044, upper bound: 1.7587780
time: 7.92 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7271344, upper bound: 1.7587776
time: 6.47 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 28.71 seconds
IS_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 28.71
Output dim: 0, lower bound: -1.7257044, upper bound: 1.7587780
IS_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 28.71
Output dim: 0, lower bound: -1.7271344, upper bound: 1.7587776

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 5.1766558, 8.2762184, 5.2358294, 8.2521811, -3.0467906, 3.0403891
1: -21.6638508, -17.3844719, -21.6125832, -17.3819675, -3.4799051, 3.4821887
2: -5.6379557, -2.5017843, -5.6968193, -2.5078504, -2.9888248, 2.9953961
3: -14.0210295, -10.9499350, -14.0002136, -10.9727945, -2.6981931, 2.7009094
4: -9.2286510, -6.2858047, -9.2266445, -6.2915306, -2.5695124, 2.5643291
5: -7.6682730, -4.8727684, -7.6786370, -4.8560133, -2.3971310, 2.4022217
6: -5.5766373, -2.8289061, -5.5660658, -2.7821691, -2.4887371, 2.4748054
7: -11.0767984, -7.2254825, -11.0751467, -7.2482886, -3.8255081, 3.7995939
8: -4.1291094, -0.9935443, -4.1049528, -1.0044413, -2.6690445, 2.6417229
9: -4.8599758, -1.7917905, -4.8372288, -1.8274822, -2.7527628, 2.7599201

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5844

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7257044, upper bound: 1.7573027
time: 17.84 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7257044, upper bound: 1.7525400
time: 10.93 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 5.1738834, 8.2792654, 5.2101388, 8.2992210, -3.0921679, 3.0691266
1: -21.6661892, -17.3821621, -21.6235275, -17.3524094, -3.5166912, 3.4945340
2: -5.6422176, -2.5007501, -5.7241244, -2.4858098, -3.0126591, 3.0336108
3: -14.0223885, -10.9374237, -14.0349970, -10.9316607, -2.7216921, 2.7334719
4: -9.2321672, -6.2843018, -9.2564821, -6.2719536, -2.5901041, 2.6047595
5: -7.6701446, -4.8707867, -7.6844544, -4.8327599, -2.4229140, 2.4107122
6: -5.5824637, -2.8284712, -5.5920420, -2.7714210, -2.5003812, 2.4932725
7: -11.0781212, -7.2060304, -11.1315813, -7.1893835, -3.8621073, 3.8551459
8: -4.1311898, -0.9895144, -4.1407986, -0.9865899, -2.6828237, 2.6835451
9: -4.8640852, -1.7905331, -4.8557568, -1.8067646, -2.7767467, 2.7818189

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 4656

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7271344, upper bound: 1.7530850
time: 8.63 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.7214667, upper bound: 1.7530855
time: 7.06 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 30.08 seconds
IS_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 30.08
Output dim: 0, lower bound: -1.7257044, upper bound: 1.7573027
IS_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 30.08
Output dim: 0, lower bound: -1.7257044, upper bound: 1.7525400
IS_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 30.08
Output dim: 0, lower bound: -1.7271344, upper bound: 1.7530850
IS_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 30.08
Output dim: 0, lower bound: -1.7214667, upper bound: 1.7530855

## BFS IS instance: IS_B1_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 5.1993685, 8.2322235, 5.2358294, 8.2521811, -3.0245566, 2.9963942
1: -21.6553383, -17.4117928, -21.6125832, -17.3819675, -3.4683199, 3.4502525
2: -5.6153574, -2.5228100, -5.6968193, -2.5078504, -2.9569645, 2.9666700
3: -13.9877892, -10.9785290, -14.0002136, -10.9727945, -2.6666451, 2.6749241
4: -9.2022018, -6.3039885, -9.2266445, -6.2915306, -2.5336761, 2.5451870
5: -7.6640301, -4.8941236, -7.6786370, -4.8560133, -2.3873940, 2.3781562
6: -5.5563807, -2.8391728, -5.5660658, -2.7821691, -2.4701042, 2.4641283
7: -11.0217352, -7.2650933, -11.0751467, -7.2482886, -3.7697811, 3.7696099
8: -4.0954981, -1.0073781, -4.1049528, -1.0044413, -2.6231070, 2.6272435
9: -4.8455315, -1.8110690, -4.8372288, -1.8274822, -2.7335625, 2.7415357

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5821
type: A, layer: 1, pos: 5821
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4644
type: B, layer: 1, pos: 4644
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 907

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7231597, upper bound: 1.7569323
time: 6.28 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_A1_B1_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.7257025, upper bound: 1.7573041
time: 11.80 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 32.52 seconds
IS_B1_A2_B2_A1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 32.52
Output dim: 0, lower bound: -1.7231597, upper bound: 1.7569323
IS_B1_A2_B2_A1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 32.52
Output dim: 0, lower bound: -1.7257025, upper bound: 1.7573041
Binary search (step 3): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.068826198577881
rel_dist={0: [-1.7599642570566427, 1.7599639862673033]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 1713.09 seconds
