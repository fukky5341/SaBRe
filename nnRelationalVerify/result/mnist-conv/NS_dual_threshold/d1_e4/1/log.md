## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.3552878565


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3612347, -6.1413960, -7.3612347, -6.1413960, -0.7229214, 0.7229214)
1: (-6.7012358, -5.3436651, -6.7012358, -5.3436651, -1.1029615, 1.1029606)
2: (-4.7904668, -3.6836765, -4.7904668, -3.6836765, -0.8028684, 0.8028684)
3: (-5.1534758, -3.8338423, -5.1534758, -3.8338423, -0.7839131, 0.7839131)
4: (-10.7349405, -9.5796747, -10.7349405, -9.5796747, -0.6515818, 0.6515818)
5: (1.3470602, 2.2432051, 1.3470602, 2.2432051, -0.6103714, 0.6103711)
6: (0.1716712, 1.3331558, 0.1716712, 1.3331558, -0.6167908, 0.6167908)
7: (-12.6086416, -11.2634993, -12.6086416, -11.2634993, -0.8833489, 0.8833489)
8: (6.0859089, 7.0352283, 6.0859089, 7.0352283, -0.8516474, 0.8516474)
9: (-8.6910915, -7.4509592, -8.6910915, -7.4509592, -1.0465469, 1.0465469)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.47 + 34.56 = 56.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3556433, upper bound: 0.3556435

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 542

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3542100
time: 5.27 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3556404
time: 4.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.95 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.95
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3542100
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.95
Output dim: 8, lower bound: -0.3556411, upper bound: 0.3556404

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.3501544, -6.1493382, -7.3549566, -6.1415854, -0.7112823, 0.7086926
1: -6.6940956, -5.3489256, -6.6975384, -5.3439026, -1.0955210, 1.0941868
2: -4.7834120, -3.6887298, -4.7866006, -3.6839380, -0.7955704, 0.7931728
3: -5.1407027, -3.8481073, -5.1527863, -3.8422637, -0.7630167, 0.7689466
4: -10.7148981, -9.5955238, -10.7228374, -9.5800114, -0.6310413, 0.6235394
5: 1.3560526, 2.2318153, 1.3473964, 2.2364359, -0.5947950, 0.5987415
6: 0.1795981, 1.3211713, 0.1718802, 1.3269817, -0.6024127, 0.6043971
7: -12.5972309, -11.2718582, -12.6022482, -11.2638721, -0.8713889, 0.8685408
8: 6.0933132, 7.0272436, 6.0863895, 7.0304656, -0.8396015, 0.8432007
9: -8.6850538, -7.4543557, -8.6883678, -7.4512520, -1.0396404, 1.0403242

Time for backsubstitution: 19.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3555321, upper bound: 0.3534478
time: 4.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556404, upper bound: 0.3542083
time: 4.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.3612232, -6.1413946, -7.3612266, -6.1413946, -0.7085247, 0.7219603
1: -6.7012305, -5.3436656, -6.7012329, -5.3436646, -1.0969591, 1.1029577
2: -4.7904620, -3.6836762, -4.7904639, -3.6836760, -0.7945137, 0.8025360
3: -5.1534753, -3.8338542, -5.1534748, -3.8338487, -0.7802844, 0.7634988
4: -10.7349195, -9.5796728, -10.7349281, -9.5796738, -0.6219268, 0.6462188
5: 1.3470609, 2.2431931, 1.3470592, 2.2431979, -0.6070225, 0.5946522
6: 0.1716713, 1.3331468, 0.1716726, 1.3331505, -0.6167150, 0.6018190
7: -12.6086330, -11.2635012, -12.6086388, -11.2634993, -0.8688607, 0.8833447
8: 6.0859103, 7.0352201, 6.0859084, 7.0352235, -0.8516431, 0.8427205
9: -8.6910868, -7.4509592, -8.6910877, -7.4509573, -1.0414033, 1.0465436

Time for backsubstitution: 20.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556370, upper bound: 0.3543109
time: 4.91 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556370, upper bound: 0.3556364
time: 4.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.82 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.82
Output dim: 8, lower bound: -0.3555321, upper bound: 0.3534478
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.82
Output dim: 8, lower bound: -0.3556404, upper bound: 0.3542083
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.82
Output dim: 8, lower bound: -0.3556370, upper bound: 0.3543109
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.82
Output dim: 8, lower bound: -0.3556370, upper bound: 0.3556364

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.3493423, -6.1511812, -7.3535380, -6.1446648, -0.7067151, 0.7046857
1: -6.6898656, -5.3498559, -6.6905231, -5.3454881, -1.0849228, 1.0818739
2: -4.7816758, -3.6894832, -4.7836914, -3.6852388, -0.7914758, 0.7884226
3: -5.1345243, -3.8491871, -5.1424108, -3.8441501, -0.7545786, 0.7573900
4: -10.7139845, -9.6027365, -10.7211990, -9.5921144, -0.6183779, 0.6153398
5: 1.3581460, 2.2309368, 1.3509250, 2.2349026, -0.5902557, 0.5937166
6: 0.1798124, 1.3192189, 0.1722441, 1.3237244, -0.5964990, 0.5996594
7: -12.5952024, -11.2732859, -12.5987282, -11.2662735, -0.8654518, 0.8619885
8: 6.0963202, 7.0263247, 6.0914569, 7.0290747, -0.8347816, 0.8369660
9: -8.6839428, -7.4605236, -8.6864719, -7.4615946, -1.0273056, 1.0314708

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542004, upper bound: 0.3534444
time: 4.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3555274, upper bound: 0.3534439
time: 6.88 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.3501511, -6.1493406, -7.3576875, -6.1401377, -0.7128811, 0.7111936
1: -6.6940894, -5.3489232, -6.6996341, -5.3419399, -1.0939388, 1.0951996
2: -4.7834072, -3.6887298, -4.7874818, -3.6808641, -0.7987452, 0.7936902
3: -5.1406898, -3.8481088, -5.1542778, -3.8337674, -0.7685592, 0.7679734
4: -10.7149000, -9.5955324, -10.7333441, -9.5792236, -0.6294484, 0.6320610
5: 1.3560562, 2.2318139, 1.3461552, 2.2410526, -0.5990949, 0.5994773
6: 0.1795980, 1.3211684, 0.1703508, 1.3277285, -0.6028261, 0.6043472
7: -12.5972271, -11.2718601, -12.6040649, -11.2631931, -0.8712382, 0.8709378
8: 6.0933189, 7.0272427, 6.0852690, 7.0350924, -0.8449268, 0.8432918
9: -8.6850529, -7.4543667, -8.6985798, -7.4503856, -1.0382385, 1.0512033

Time for backsubstitution: 21.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: A, layer: 1, pos: 6182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556364, upper bound: 0.3528809
time: 4.67 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556364, upper bound: 0.3542042
time: 4.69 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.3612232, -6.1413946, -7.3609943, -6.1416121, -0.7078648, 0.7215445
1: -6.7012305, -5.3436656, -6.6994863, -5.3438854, -1.0957127, 1.1001282
2: -4.7904620, -3.6836762, -4.7901211, -3.6868272, -0.7913527, 0.8022642
3: -5.1534753, -3.8338542, -5.1475821, -3.8339796, -0.7801526, 0.7575123
4: -10.7349195, -9.5796728, -10.7343845, -9.5802174, -0.6213701, 0.6456826
5: 1.3470609, 2.2431931, 1.3475761, 2.2419541, -0.6057715, 0.5941694
6: 0.1716713, 1.3331468, 0.1718681, 1.3298290, -0.6131210, 0.6015320
7: -12.6086330, -11.2635012, -12.6077080, -11.2637386, -0.8686066, 0.8823514
8: 6.0859103, 7.0352201, 6.0876474, 7.0350513, -0.8503952, 0.8402810
9: -8.6910868, -7.4509592, -8.6908741, -7.4531193, -1.0390692, 1.0464706

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3542002
time: 5.99 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556361, upper bound: 0.3543104
time: 4.42 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.3612227, -6.1413956, -7.3677430, -6.1346502, -0.7191343, 0.7243402
1: -6.7012300, -5.3436642, -6.7119431, -5.3364639, -1.1117768, 1.1134453
2: -4.7904596, -3.6836805, -4.8245387, -3.6773720, -0.8030648, 0.8098526
3: -5.1534672, -3.8338530, -5.1613183, -3.7868261, -0.7827039, 0.7718980
4: -10.7349195, -9.5796738, -10.7413759, -9.5719261, -0.6298006, 0.6549895
5: 1.3470621, 2.2431898, 1.3328304, 2.2450576, -0.6089549, 0.6028829
6: 0.1716723, 1.3331410, 0.1390129, 1.3371429, -0.6224573, 0.6106179
7: -12.6086292, -11.2634983, -12.6128626, -11.2527885, -0.8802142, 0.8881121
8: 6.0859127, 7.0352197, 6.0788851, 7.0395288, -0.8532944, 0.8566818
9: -8.6910858, -7.4509616, -8.7193203, -7.4472160, -1.0446720, 1.0710754

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3555279
time: 4.95 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3556361, upper bound: 0.3556357
time: 4.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.70 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3542004, upper bound: 0.3534444
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3555274, upper bound: 0.3534439
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3556364, upper bound: 0.3528809
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3556364, upper bound: 0.3542042
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3542002
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3556361, upper bound: 0.3543104
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3555279
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.70
Output dim: 8, lower bound: -0.3556361, upper bound: 0.3556357

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3558397, -6.1443982, -7.3535366, -6.1446629, -0.7130327, 0.7152054
1: -6.7006707, -5.3426580, -6.6905193, -5.3454885, -1.0956402, 1.0961394
2: -4.8157816, -3.6831784, -4.7836914, -3.6852436, -0.8031063, 0.7969165
3: -5.1423430, -3.8021536, -5.1424031, -3.8441491, -0.7629027, 0.7661295
4: -10.7204027, -9.5950089, -10.7212000, -9.5921154, -0.6271555, 0.6232362
5: 1.3439088, 2.2328403, 1.3509259, 2.2349000, -0.5963681, 0.5956836
6: 0.1471508, 1.3232573, 0.1722441, 1.3237178, -0.6037331, 0.6085835
7: -12.5995140, -11.2625666, -12.5987244, -11.2662754, -0.8703213, 0.8715544
8: 6.0893421, 7.0306358, 6.0914602, 7.0290756, -0.8475966, 0.8400617
9: -8.7121277, -7.4568186, -8.6864719, -7.4615974, -1.0544376, 1.0346985

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: A, layer: 1, pos: 4614

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3548751, upper bound: 0.3534444
time: 5.03 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3548750, upper bound: 0.3534444
time: 4.62 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -7.3501511, -6.1493406, -7.3574567, -6.1403513, -0.7122173, 0.7107830
1: -6.6940894, -5.3489232, -6.6978917, -5.3421617, -1.0926876, 1.0923729
2: -4.7834072, -3.6887298, -4.7871423, -3.6840110, -0.7955904, 0.7934189
3: -5.1406898, -3.8481088, -5.1483836, -3.8338966, -0.7684269, 0.7619853
4: -10.7149000, -9.5955324, -10.7328033, -9.5797682, -0.6288929, 0.6315286
5: 1.3560562, 2.2318139, 1.3466725, 2.2398090, -0.5978432, 0.5989919
6: 0.1795980, 1.3211684, 0.1705478, 1.3244109, -0.5992324, 0.6040623
7: -12.5972271, -11.2718601, -12.6031342, -11.2634287, -0.8709855, 0.8699441
8: 6.0933189, 7.0272427, 6.0870214, 7.0349202, -0.8436799, 0.8408527
9: -8.6850529, -7.4543667, -8.6983624, -7.4525452, -1.0359030, 1.0511284

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: A, layer: 1, pos: 6182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543096, upper bound: 0.3528803
time: 6.92 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543096, upper bound: 0.3528800
time: 7.23 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -7.3501530, -6.1493406, -7.3642230, -6.1333466, -0.7234969, 0.7167220
1: -6.6940861, -5.3489246, -6.7104368, -5.3347335, -1.1087923, 1.1056633
2: -4.7834086, -3.6887341, -4.8215923, -3.6745505, -0.8073215, 0.8048708
3: -5.1406803, -3.8481102, -5.1620927, -3.7867746, -0.7710290, 0.7763491
4: -10.7148981, -9.5955315, -10.7397852, -9.5714808, -0.6373439, 0.6407323
5: 1.3560570, 2.2318106, 1.3319297, 2.2429044, -0.6010151, 0.6079843
6: 0.1795987, 1.3211619, 0.1376883, 1.3317680, -0.6117172, 0.6123178
7: -12.5972252, -11.2718639, -12.6082764, -11.2524834, -0.8829942, 0.8756900
8: 6.0933218, 7.0272436, 6.0782690, 7.0394030, -0.8472323, 0.8580346
9: -8.6850538, -7.4543715, -8.7268763, -7.4466429, -1.0415101, 1.0701103

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: A, layer: 1, pos: 6182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542047, upper bound: 0.3542042
time: 6.44 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3542047, upper bound: 0.3542042
time: 5.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.3639517, -6.1399512, -7.3609934, -6.1416140, -0.7103643, 0.7231462
1: -6.7033253, -5.3417029, -6.6994801, -5.3438869, -1.0967312, 1.0985508
2: -4.7913442, -3.6806018, -4.7901177, -3.6868277, -0.7918611, 0.8052063
3: -5.1549630, -3.8253508, -5.1475677, -3.8339810, -0.7791836, 0.7654629
4: -10.7454290, -9.5788889, -10.7343836, -9.5802259, -0.6326928, 0.6440766
5: 1.3458242, 2.2478123, 1.3475804, 2.2419519, -0.6065090, 0.5984704
6: 0.1701412, 1.3338964, 0.1718671, 1.3298275, -0.6122627, 0.6019468
7: -12.6104527, -11.2628241, -12.6077061, -11.2637405, -0.8710065, 0.8821988
8: 6.0847788, 7.0398459, 6.0876513, 7.0350509, -0.8504839, 0.8456068
9: -8.7013025, -7.4500933, -8.6908712, -7.4531312, -1.0499506, 1.0450673

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 6182

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543093, upper bound: 0.3543102
time: 5.45 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3543093, upper bound: 0.3543097
time: 6.57 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3598070, -6.1444736, -7.3669181, -6.1364846, -0.7151289, 0.7197521
1: -6.6942201, -5.3452568, -6.7077298, -5.3373938, -1.0990958, 1.1029882
2: -4.7875571, -3.6849825, -4.8227863, -3.6781387, -0.7982717, 0.8055456
3: -5.1430860, -3.8357356, -5.1551394, -3.7879090, -0.7711585, 0.7634373
4: -10.7332792, -9.5917807, -10.7404461, -9.5791378, -0.6215971, 0.6422105
5: 1.3505893, 2.2416575, 1.3349180, 2.2442057, -0.6039388, 0.5979648
6: 0.1720344, 1.3298913, 0.1392235, 1.3351895, -0.6174059, 0.6044490
7: -12.6051140, -11.2659025, -12.6108818, -11.2542114, -0.8733120, 0.8822289
8: 6.0909834, 7.0338297, 6.0818996, 7.0386114, -0.8470116, 0.8513432
9: -8.6891947, -7.4613056, -8.7182055, -7.4533882, -1.0358176, 1.0587430

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3548755
time: 5.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3555279
time: 4.46 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.3639531, -6.1399498, -7.3677435, -6.1346531, -0.7216187, 0.7259405
1: -6.7033215, -5.3417044, -6.7119350, -5.3364649, -1.1109977, 1.1118660
2: -4.7913427, -3.6806076, -4.8245339, -3.6773713, -0.8035750, 0.8120944
3: -5.1549535, -3.8253527, -5.1613059, -3.7868278, -0.7817345, 0.7797971
4: -10.7454300, -9.5788889, -10.7413750, -9.5719357, -0.6382465, 0.6533823
5: 1.3458247, 2.2478085, 1.3328333, 2.2450559, -0.6096919, 0.6048784
6: 0.1701437, 1.3338897, 0.1390135, 1.3371391, -0.6213326, 0.6097388
7: -12.6104498, -11.2628231, -12.6128578, -11.2527905, -0.8810625, 0.8879609
8: 6.0847807, 7.0398474, 6.0788898, 7.0395279, -0.8534069, 0.8586383
9: -8.7013025, -7.4500971, -8.7193193, -7.4472284, -1.0555544, 1.0696778

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 6182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3542042, upper bound: 0.3556344
time: 7.32 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3542042, upper bound: 0.3556358
time: 5.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 35.16 seconds
NS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3548751, upper bound: 0.3534444
NS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3548750, upper bound: 0.3534444
NS_A1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3543096, upper bound: 0.3528803
NS_A1_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3543096, upper bound: 0.3528800
NS_A1_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3542047, upper bound: 0.3542042
NS_A1_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3542047, upper bound: 0.3542042
NS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3543093, upper bound: 0.3543102
NS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3543093, upper bound: 0.3543097
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3548755
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3548757, upper bound: 0.3555279
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3542042, upper bound: 0.3556344
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.16
Output dim: 8, lower bound: -0.3542042, upper bound: 0.3556358

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3598070, -6.1444736, -7.3704834, -6.1331506, -0.7187748, 0.7217975
1: -6.6942201, -5.3452568, -6.7141256, -5.3344955, -1.1003213, 1.1090374
2: -4.7875571, -3.6849825, -4.8254352, -3.6742930, -0.8022113, 0.8076630
3: -5.1430860, -3.8357356, -5.1628008, -3.7783678, -0.7743039, 0.7709975
4: -10.7332792, -9.5917807, -10.7518768, -9.5711432, -0.6283216, 0.6458352
5: 1.3505893, 2.2416575, 1.3315992, 2.2496643, -0.6072643, 0.5993679
6: 0.1720344, 1.3298913, 0.1374822, 1.3379210, -0.6194074, 0.6046743
7: -12.6051140, -11.2659025, -12.6146584, -11.2521162, -0.8742404, 0.8859344
8: 6.0909834, 7.0338297, 6.0777502, 7.0441599, -0.8499360, 0.8538628
9: -8.6891947, -7.4613056, -8.7296085, -7.4463301, -1.0432525, 1.0634022

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3534438, upper bound: 0.3555266
time: 8.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3534438, upper bound: 0.3555274
time: 6.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3639531, -6.1399498, -7.3566713, -6.1425662, -0.7207153, 0.7144604
1: -6.7033215, -5.3417044, -6.7048860, -5.3417268, -1.1086535, 1.1047382
2: -4.7913427, -3.6806076, -4.8175383, -3.6824119, -0.8064556, 0.8051522
3: -5.1549535, -3.8253527, -5.1485052, -3.8010693, -0.7674775, 0.7769077
4: -10.7454300, -9.5788889, -10.7213316, -9.5878067, -0.6366086, 0.6331561
5: 1.3458247, 2.2478085, 1.3418226, 2.2336946, -0.5983102, 0.6033523
6: 0.1701437, 1.3338897, 0.1469394, 1.3252152, -0.6091933, 0.6091120
7: -12.6104498, -11.2628231, -12.6015034, -11.2611446, -0.8795466, 0.8764710
8: 6.0847807, 7.0398474, 6.0863371, 7.0315514, -0.8454099, 0.8549557
9: -8.7013025, -7.4500971, -8.7132530, -7.4506602, -1.0572033, 1.0631623

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 6182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3556345
time: 5.15 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3543090
time: 7.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.3639531, -6.1399498, -7.3677368, -6.1346521, -0.7214525, 0.7164350
1: -6.7033215, -5.3417044, -6.7119355, -5.3364668, -1.1106815, 1.1058683
2: -4.7913427, -3.6806076, -4.8245330, -3.6773720, -0.8035746, 0.8100960
3: -5.1549535, -3.8253527, -5.1613064, -3.7868340, -0.7778664, 0.7797608
4: -10.7454300, -9.5788889, -10.7413673, -9.5719366, -0.6382463, 0.6292210
5: 1.3458247, 2.2478085, 1.3328340, 2.2450526, -0.5973155, 0.6047606
6: 0.1701437, 1.3338897, 0.1390119, 1.3371360, -0.6108191, 0.6097386
7: -12.6104498, -11.2628231, -12.6128540, -11.2527924, -0.8808961, 0.8734746
8: 6.0847807, 7.0398474, 6.0788918, 7.0395241, -0.8459010, 0.8582864
9: -8.7013025, -7.4500971, -8.7193155, -7.4472294, -1.0555539, 1.0679977

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614
type: B, layer: 1, pos: 6182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3556351
time: 7.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3543451
time: 6.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 37.38 seconds
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 37.38
Output dim: 8, lower bound: -0.3534438, upper bound: 0.3555266
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 37.38
Output dim: 8, lower bound: -0.3534438, upper bound: 0.3555274
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 37.38
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3556345
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 37.38
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3543090
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 37.38
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3556351
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 37.38
Output dim: 8, lower bound: -0.3528800, upper bound: 0.3543451

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -7.3598070, -6.1444736, -7.3594170, -6.1410923, -0.7180166, 0.7103190
1: -6.6942201, -5.3452568, -6.7070680, -5.3397522, -1.0979686, 1.1018953
2: -4.7875571, -3.6849825, -4.8184247, -3.6793323, -0.8044500, 0.8007112
3: -5.1430860, -3.8357356, -5.1499996, -3.7926083, -0.7600436, 0.7686872
4: -10.7332792, -9.5917807, -10.7318296, -9.5870152, -0.6266823, 0.6256208
5: 1.3505893, 2.2416575, 1.3405836, 2.2383022, -0.5958803, 0.5978353
6: 0.1720344, 1.3298913, 0.1454070, 1.3259825, -0.6072450, 0.6040559
7: -12.6051140, -11.2659025, -12.6033030, -11.2604656, -0.8727117, 0.8744323
8: 6.0909834, 7.0338297, 6.0852251, 7.0361838, -0.8419394, 0.8501716
9: -8.6891947, -7.4613056, -8.7235317, -7.4497590, -1.0449042, 1.0568681

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 6212
type: B, layer: 1, pos: 6212
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4614
type: B, layer: 1, pos: 4614

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3521231, upper bound: 0.3555272
time: 4.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3521231, upper bound: 0.3541994
time: 8.96 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -7.3598070, -6.1444736, -7.3704796, -6.1331525, -0.7187676, 0.7139072
1: -6.6942201, -5.3452568, -6.7141247, -5.3344936, -1.1000061, 1.1030378
2: -4.7875571, -3.6849825, -4.8254323, -3.6742938, -0.8022113, 0.8048723
3: -5.1430860, -3.8357356, -5.1628008, -3.7783728, -0.7706747, 0.7709975
4: -10.7332792, -9.5917807, -10.7518702, -9.5711432, -0.6283214, 0.6301322
5: 1.3505893, 2.2416575, 1.3316002, 2.2496593, -0.5970702, 0.5992508
6: 0.1720344, 1.3298913, 0.1374800, 1.3379185, -0.6087728, 0.6046746
7: -12.6051140, -11.2659025, -12.6146536, -11.2521143, -0.8740749, 0.8714509
8: 6.0909834, 7.0338297, 6.0777497, 7.0441556, -0.8458853, 0.8535137
9: -8.6891947, -7.4613056, -8.7296066, -7.4463291, -1.0432520, 1.0617323

Time for backsubstitution: 20.97 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.03 + 563.96 = 619.98 seconds
