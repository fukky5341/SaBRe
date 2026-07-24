## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0301740019999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2329988, 3.2329993)
1: (-12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8305802, 2.8305807)
2: (-5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056412, 2.8056412)
3: (-5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5602217, 3.5602221)
4: (-11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3337431, 3.3337431)
5: (-6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6769562, 2.6769567)
6: (-12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9359779, 2.9359775)
7: (-8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884)
8: (7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2254882, 2.2254882)
9: (-6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9353604, 2.9353604)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.39 + 34.56 = 57.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.0405798, upper bound: 1.0405789

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392414, upper bound: 1.0405805
time: 9.49 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405779, upper bound: 1.0405770
time: 4.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.19
Output dim: 8, lower bound: -1.0392414, upper bound: 1.0405805
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.19
Output dim: 8, lower bound: -1.0405779, upper bound: 1.0405770

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.1548772, -2.0182886, -6.1859760, -2.0057139, -3.1569152, 3.2016006
1: -12.2045307, -9.0081415, -12.2218876, -8.9927692, -2.7952924, 2.8191171
2: -5.5950623, -2.2592859, -5.6188431, -2.2436464, -2.7779942, 2.8107114
3: -5.3431683, -1.5320258, -5.3683939, -1.5167837, -3.5291557, 3.5204468
4: -11.4873409, -7.5901594, -11.5171738, -7.5781326, -3.1162300, 3.3105268
5: -6.2218180, -3.1273403, -6.2568622, -3.0843782, -2.5984764, 2.5789490
6: -12.3938379, -8.7286739, -12.4124298, -8.6987143, -2.8914933, 2.8549919
7: -8.1438246, -4.6864681, -8.1612873, -4.6738949, -3.4278345, 3.4748192
8: 7.7507830, 10.0393200, 7.7418718, 10.0556536, -2.1846361, 2.1970518
9: -6.3052931, -2.8363342, -6.3405223, -2.8203778, -2.8691535, 2.8869257

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383414, upper bound: 1.0405791
time: 8.07 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392400, upper bound: 1.0405763
time: 5.50 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.1948948, -1.9940524, -6.1948996, -1.9940438, -3.2329769, 3.2234554
1: -12.2345190, -8.9912539, -12.2345247, -8.9912539, -2.8083191, 2.8305674
2: -5.6206818, -2.2354579, -5.6206818, -2.2354555, -2.8121109, 2.8028669
3: -5.3708839, -1.5042260, -5.3708878, -1.5042186, -3.5584278, 3.5457082
4: -11.5237923, -7.5678382, -11.5237989, -7.5678282, -3.3288918, 3.3186941
5: -6.2900391, -3.0817842, -6.2900629, -3.0817819, -2.6503735, 2.6744058
6: -12.4278564, -8.6957150, -12.4278679, -8.6957130, -2.9230051, 2.9345174
7: -8.1703444, -4.6722722, -8.1703491, -4.6722727, -3.4980717, 3.4980769
8: 7.7388439, 10.0601463, 7.7388420, 10.0601492, -2.2225485, 2.2251823
9: -6.3480139, -2.8028798, -6.3480167, -2.8028655, -2.9353271, 2.9207177

Time for backsubstitution: 22.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396744, upper bound: 1.0405791
time: 5.22 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405766, upper bound: 1.0405756
time: 4.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.66 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 32.66
Output dim: 8, lower bound: -1.0383414, upper bound: 1.0405791
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 32.66
Output dim: 8, lower bound: -1.0392400, upper bound: 1.0405763
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.66
Output dim: 8, lower bound: -1.0396744, upper bound: 1.0405791
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.66
Output dim: 8, lower bound: -1.0405766, upper bound: 1.0405756

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.1432724, -2.0185342, -6.1637802, -2.0064535, -3.1444120, 3.1791315
1: -12.1966906, -9.0084362, -12.2067089, -8.9933357, -2.7868547, 2.8035984
2: -5.5853491, -2.2610905, -5.6002512, -2.2474942, -2.7647190, 2.7901587
3: -5.3422060, -1.5337963, -5.3665223, -1.5204852, -3.5188103, 3.5119472
4: -11.4855919, -7.5956421, -11.5123711, -7.5886278, -3.1040106, 3.3005843
5: -6.2206297, -3.1290174, -6.2545671, -3.0877032, -2.5869694, 2.5687680
6: -12.3928843, -8.7296352, -12.4105654, -8.7006855, -2.8789692, 2.8433032
7: -8.1415615, -4.6900539, -8.1559687, -4.6807652, -3.4168186, 3.4659147
8: 7.7520466, 10.0366726, 7.7444558, 10.0505981, -2.1729589, 2.1862919
9: -6.3034601, -2.8374925, -6.3370152, -2.8226132, -2.8610263, 2.8782716

Time for backsubstitution: 22.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369561, upper bound: 1.0400848
time: 5.80 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383398, upper bound: 1.0405755
time: 4.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.1548529, -2.0182884, -6.1947799, -1.9594219, -3.1959143, 3.2019339
1: -12.2045097, -9.0081425, -12.2343206, -8.9759340, -2.8127894, 2.8311410
2: -5.5950441, -2.2592912, -5.6210656, -2.2065835, -2.8097639, 2.8045058
3: -5.3431664, -1.5320287, -5.3838110, -1.5132725, -3.5351963, 3.5304661
4: -11.4873390, -7.5901666, -11.5400257, -7.5723209, -3.1184883, 3.3331008
5: -6.2218175, -3.1273441, -6.2692909, -3.0820622, -2.6064034, 2.5879784
6: -12.3938351, -8.7286777, -12.4236259, -8.6965933, -2.9023495, 2.8591566
7: -8.1438198, -4.6864719, -8.1785889, -4.6735711, -3.4270668, 3.4921169
8: 7.7507839, 10.0393124, 7.7333207, 10.0636082, -2.1895456, 2.2121007
9: -6.3052883, -2.8363369, -6.3434873, -2.8053327, -2.8877573, 2.8869295

Time for backsubstitution: 22.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378022, upper bound: 1.0400820
time: 6.83 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392385, upper bound: 1.0405756
time: 7.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.1833057, -1.9944351, -6.1727118, -1.9947834, -3.2204876, 3.2008944
1: -12.2265892, -8.9915485, -12.2193756, -8.9918175, -2.7996569, 2.8150229
2: -5.6109905, -2.2374520, -5.6021366, -2.2392869, -2.7988596, 2.7822680
3: -5.3699121, -1.5061595, -5.3690143, -1.5079060, -3.5481176, 3.5362277
4: -11.5212803, -7.5733180, -11.5189610, -7.5783176, -3.3160162, 3.3087349
5: -6.2888441, -3.0835323, -6.2877684, -3.0851078, -2.6388578, 2.6633220
6: -12.4268827, -8.6967468, -12.4259968, -8.6976986, -2.9104276, 2.9218912
7: -8.1675720, -4.6758604, -8.1650305, -4.6791420, -3.4884300, 3.4891701
8: 7.7401853, 10.0575047, 7.7414179, 10.0551004, -2.2103562, 2.2144029
9: -6.3461843, -2.8040371, -6.3445220, -2.8050938, -2.9272046, 2.9120150

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382986, upper bound: 1.0400816
time: 4.84 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396729, upper bound: 1.0405777
time: 5.07 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.1948690, -1.9940529, -6.2037091, -1.9477553, -3.2584591, 3.2243056
1: -12.2345009, -8.9912548, -12.2469406, -8.9744186, -2.8258724, 2.8427114
2: -5.6206627, -2.2354631, -5.6229372, -2.1984115, -2.8210273, 2.7969818
3: -5.3708830, -1.5042317, -5.3863239, -1.5007055, -3.5645676, 3.5553427
4: -11.5237865, -7.5678458, -11.5466595, -7.5620203, -3.3332872, 3.3413768
5: -6.2900367, -3.0817871, -6.3024697, -3.0794792, -2.6583118, 2.6782780
6: -12.4278545, -8.6957178, -12.4390335, -8.6935816, -2.9314737, 2.9356523
7: -8.1703386, -4.6722789, -8.1877117, -4.6719484, -3.4983902, 3.5154328
8: 7.7388477, 10.0601397, 7.7302904, 10.0681124, -2.2275724, 2.2402234
9: -6.3480101, -2.8028789, -6.3509536, -2.7878308, -2.9539185, 2.9206514

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391533, upper bound: 1.0400819
time: 5.04 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405750, upper bound: 1.0405740
time: 4.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.82 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0369561, upper bound: 1.0400848
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0383398, upper bound: 1.0405755
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0378022, upper bound: 1.0400820
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0392385, upper bound: 1.0405756
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0382986, upper bound: 1.0400816
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0396729, upper bound: 1.0405777
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0391533, upper bound: 1.0400819
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.82
Output dim: 8, lower bound: -1.0405750, upper bound: 1.0405740

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.1327419, -2.0422616, -6.1621170, -2.0178668, -3.1197319, 3.1533761
1: -12.1830835, -9.0147772, -12.2011642, -8.9958467, -2.7705688, 2.7908759
2: -5.5792904, -2.2820330, -5.5989838, -2.2571483, -2.7474976, 2.7669511
3: -5.3238859, -1.5427182, -5.3585811, -1.5227036, -3.4972992, 3.4941487
4: -11.4700356, -7.6044531, -11.5061731, -7.5907454, -3.0858269, 3.2838459
5: -6.2119541, -3.1503296, -6.2530689, -3.0978904, -2.5664301, 2.5444646
6: -12.3850632, -8.7424889, -12.4083443, -8.7066278, -2.8651233, 2.8278437
7: -8.1228828, -4.6933870, -8.1483126, -4.6818976, -3.3953543, 3.4549255
8: 7.7636724, 10.0293818, 7.7496834, 10.0482483, -2.1584005, 2.1706078
9: -6.2918553, -2.8501053, -6.3343048, -2.8277166, -2.8443141, 2.8621044

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369521, upper bound: 1.0392694
time: 7.18 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369521, upper bound: 1.0400831
time: 5.68 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.1432714, -2.0185392, -6.1637812, -2.0064578, -3.1444077, 3.1617866
1: -12.1966858, -9.0084343, -12.2067108, -8.9933348, -2.7867446, 2.8068075
2: -5.5853491, -2.2610936, -5.6002512, -2.2474928, -2.7647181, 2.7855191
3: -5.3422022, -1.5337956, -5.3665228, -1.5204849, -3.5109882, 3.5119457
4: -11.4855881, -7.5956445, -11.5123730, -7.5886278, -3.1017303, 3.3005810
5: -6.2206278, -3.1290226, -6.2545686, -3.0877051, -2.5869665, 2.5527034
6: -12.3928823, -8.7296381, -12.4105644, -8.7006893, -2.8793221, 2.8428702
7: -8.1415596, -4.6900549, -8.1559668, -4.6807647, -3.4218998, 3.4659119
8: 7.7520504, 10.0366716, 7.7444563, 10.0505981, -2.1728573, 2.1894667
9: -6.3034582, -2.8374946, -6.3370152, -2.8226135, -2.8610239, 2.8741145

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383384, upper bound: 1.0398418
time: 6.56 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383384, upper bound: 1.0405731
time: 4.83 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.1443181, -2.0420184, -6.1931171, -1.9708116, -3.1673779, 3.1761756
1: -12.1909227, -9.0144768, -12.2288074, -8.9784546, -2.7965078, 2.8184342
2: -5.5889797, -2.2802265, -5.6197844, -2.2162671, -2.7888103, 2.7812781
3: -5.3248448, -1.5409503, -5.3759375, -1.5154920, -3.5137138, 3.5127325
4: -11.4718189, -7.5989771, -11.5338573, -7.5744605, -3.1003523, 3.3163900
5: -6.2131386, -3.1486382, -6.2677898, -3.0922213, -2.5858927, 2.5636940
6: -12.3860159, -8.7415257, -12.4214020, -8.7025166, -2.8885317, 2.8436861
7: -8.1251488, -4.6898060, -8.1709766, -4.6747031, -3.4055519, 3.4811707
8: 7.7624116, 10.0320110, 7.7385387, 10.0612650, -2.1749582, 2.1964402
9: -6.2936745, -2.8489594, -6.3407722, -2.8103249, -2.8711491, 2.8707433

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378006, upper bound: 1.0392718
time: 5.13 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378006, upper bound: 1.0400800
time: 5.84 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.1548510, -2.0182929, -6.1947808, -1.9594245, -3.1909161, 3.1846075
1: -12.2045069, -9.0081444, -12.2343187, -8.9759350, -2.8126812, 2.8343492
2: -5.5950451, -2.2592936, -5.6210642, -2.2065854, -2.8070297, 2.7998629
3: -5.3431640, -1.5320296, -5.3838091, -1.5132730, -3.5273752, 3.5304642
4: -11.4873371, -7.5901690, -11.5400276, -7.5723205, -3.1161480, 3.3330998
5: -6.2218142, -3.1273484, -6.2692900, -3.0820622, -2.6064014, 2.5719128
6: -12.3938370, -8.7286787, -12.4236279, -8.6965923, -2.9027028, 2.8587232
7: -8.1438160, -4.6864719, -8.1785870, -4.6735706, -3.4320784, 3.4921150
8: 7.7507868, 10.0393114, 7.7333217, 10.0636063, -2.1894436, 2.2152765
9: -6.3052874, -2.8363383, -6.3434882, -2.8053334, -2.8877544, 2.8827748

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392369, upper bound: 1.0398421
time: 8.69 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392370, upper bound: 1.0405733
time: 8.04 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.1727057, -2.0183778, -6.1710296, -2.0061893, -3.1957173, 3.1748614
1: -12.2128639, -8.9978971, -12.2138529, -8.9943314, -2.7831416, 2.8022585
2: -5.6049042, -2.2587113, -5.6008639, -2.2489367, -2.7785044, 2.7590170
3: -5.3515301, -1.5153654, -5.3610706, -1.5101333, -3.5266094, 3.5182848
4: -11.5045528, -7.5820918, -11.5127707, -7.5804362, -3.2964487, 3.2920656
5: -6.2801886, -3.1049519, -6.2862706, -3.0952921, -2.6183443, 2.6389098
6: -12.4190798, -8.7096939, -12.4237766, -8.7036362, -2.8965101, 2.9063139
7: -8.1480932, -4.6792116, -8.1573973, -4.6802793, -3.4678140, 3.4781857
8: 7.7519531, 10.0501709, 7.7466373, 10.0527334, -2.1954579, 2.1986876
9: -6.3345046, -2.8166785, -6.3417830, -2.8101947, -2.9104042, 2.8957796

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382970, upper bound: 1.0392681
time: 4.63 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382970, upper bound: 1.0400804
time: 4.63 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.1833038, -1.9944425, -6.1727123, -1.9947858, -3.2204852, 3.1832523
1: -12.2265882, -8.9915504, -12.2193747, -8.9918175, -2.7994838, 2.8182316
2: -5.6109896, -2.2374578, -5.6021352, -2.2392895, -2.7967439, 2.7796617
3: -5.3699093, -1.5061588, -5.3690133, -1.5079060, -3.5402966, 3.5362244
4: -11.5212774, -7.5733166, -11.5189600, -7.5783195, -3.3136749, 3.3087335
5: -6.2888432, -3.0835361, -6.2877665, -3.0851083, -2.6388569, 2.6477313
6: -12.4268837, -8.6967516, -12.4259968, -8.6976986, -2.9107790, 2.9211650
7: -8.1675701, -4.6758595, -8.1650305, -4.6791420, -3.4884281, 3.4891710
8: 7.7401881, 10.0575027, 7.7414188, 10.0550985, -2.2101846, 2.2175779
9: -6.3461823, -2.8040409, -6.3445215, -2.8050940, -2.9272032, 2.9079914

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396713, upper bound: 1.0398414
time: 8.11 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396713, upper bound: 1.0405761
time: 5.21 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.1842642, -2.0179889, -6.2020216, -1.9591389, -3.2298150, 3.1982732
1: -12.2207937, -8.9975948, -12.2414541, -8.9769373, -2.8093319, 2.8300271
2: -5.6145735, -2.2567124, -5.6216540, -2.2080832, -2.8000884, 2.7737088
3: -5.3524990, -1.5134375, -5.3784451, -1.5029330, -3.5430822, 3.5374594
4: -11.5071115, -7.5766206, -11.5404968, -7.5641570, -3.3138008, 3.3247743
5: -6.2813764, -3.1031914, -6.3009648, -3.0896363, -2.6378269, 2.6538982
6: -12.4200497, -8.7086544, -12.4368114, -8.6995049, -2.9149828, 2.9200702
7: -8.1508808, -4.6756306, -8.1801233, -4.6730852, -3.4777956, 3.5044928
8: 7.7506132, 10.0527973, 7.7354999, 10.0657568, -2.2126570, 2.2245340
9: -6.3363194, -2.8155303, -6.3481960, -2.7928197, -2.9372244, 2.9043913

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391517, upper bound: 1.0392687
time: 5.18 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391517, upper bound: 1.0400806
time: 4.92 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.1948657, -1.9940577, -6.2037082, -1.9477553, -3.2534389, 3.2066855
1: -12.2344999, -8.9912548, -12.2469416, -8.9744177, -2.8256993, 2.8459196
2: -5.6206632, -2.2354662, -5.6229377, -2.1984124, -2.8182893, 2.7943735
3: -5.3708801, -1.5042298, -5.3863220, -1.5007069, -3.5567436, 3.5553408
4: -11.5237856, -7.5678468, -11.5466557, -7.5620208, -3.3309469, 3.3413744
5: -6.2900362, -3.0817924, -6.3024693, -3.0794821, -2.6583090, 2.6626897
6: -12.4278517, -8.6957207, -12.4390335, -8.6935825, -2.9295111, 2.9349217
7: -8.1703358, -4.6722794, -8.1877098, -4.6719494, -3.4983864, 3.5154305
8: 7.7388501, 10.0601387, 7.7302918, 10.0681133, -2.2274022, 2.2434003
9: -6.3480086, -2.8028820, -6.3509521, -2.7878301, -2.9539165, 2.9166279

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405735, upper bound: 1.0398409
time: 5.23 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405735, upper bound: 1.0405724
time: 4.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.59 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0369521, upper bound: 1.0392694
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0369521, upper bound: 1.0400831
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0383384, upper bound: 1.0398418
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0383384, upper bound: 1.0405731
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0378006, upper bound: 1.0392718
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0378006, upper bound: 1.0400800
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0392369, upper bound: 1.0398421
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0392370, upper bound: 1.0405733
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0382970, upper bound: 1.0392681
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0382970, upper bound: 1.0400804
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0396713, upper bound: 1.0398414
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0396713, upper bound: 1.0405761
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0391517, upper bound: 1.0392687
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0391517, upper bound: 1.0400806
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0405735, upper bound: 1.0398409
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.59
Output dim: 8, lower bound: -1.0405735, upper bound: 1.0405724

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.1302958, -2.0425751, -6.1550522, -2.0189395, -3.1143117, 3.1449680
1: -12.1813889, -9.0150671, -12.1962337, -8.9966831, -2.7660718, 2.7836156
2: -5.5787444, -2.2849569, -5.5974360, -2.2658286, -2.7375088, 2.7616210
3: -5.3231125, -1.5443981, -5.3563142, -1.5277669, -3.4871883, 3.4866714
4: -11.4680490, -7.6098146, -11.4996138, -7.6062951, -3.0671043, 3.2719297
5: -6.2099805, -3.1508803, -6.2473259, -3.0995283, -2.5615053, 2.5377889
6: -12.3845358, -8.7429256, -12.4068213, -8.7079277, -2.8615947, 2.8243423
7: -8.1216850, -4.6961131, -8.1442289, -4.6897783, -3.3844757, 3.4481158
8: 7.7648115, 10.0285273, 7.7530642, 10.0457554, -2.1536975, 2.1656296
9: -6.2910037, -2.8524206, -6.3318319, -2.8344300, -2.8364849, 2.8571849

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361880, upper bound: 1.0392687
time: 5.39 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361906, upper bound: 1.0392692
time: 4.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.1327238, -2.0422659, -6.1713877, -1.9981706, -3.1375570, 3.1637869
1: -12.1830740, -9.0147800, -12.2153587, -8.9916697, -2.7732487, 2.8117199
2: -5.5792861, -2.2820568, -5.6161718, -2.2512164, -2.7530737, 2.7825613
3: -5.3238826, -1.5427330, -5.3753357, -1.5165081, -3.5053005, 3.5142627
4: -11.4700193, -7.6044664, -11.5695190, -7.5869141, -3.0910196, 3.3379583
5: -6.2119436, -3.1503310, -6.2622290, -3.0859308, -2.5775208, 2.5533624
6: -12.3850584, -8.7424908, -12.4133253, -8.7033815, -2.8693538, 2.8343492
7: -8.1228752, -4.6933975, -8.1877451, -4.6780643, -3.3985434, 3.4943476
8: 7.7636819, 10.0293770, 7.7415686, 10.0626945, -2.1754847, 2.1798418
9: -6.2918491, -2.8501148, -6.3633547, -2.8249266, -2.8454432, 2.8907590

Time for backsubstitution: 21.92 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.96 + 547.48 = 605.43 seconds
