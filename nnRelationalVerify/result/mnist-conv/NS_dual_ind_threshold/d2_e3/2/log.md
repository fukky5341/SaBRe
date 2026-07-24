## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.407698893


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5688004, 0.5688006)
1: (-19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9268413, 0.9268413)
2: (-4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.9019241, 0.9019244)
3: (-11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7126610, 0.7126610)
4: (-11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8338060, 0.8338060)
5: (-7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7609878, 0.7609875)
6: (-4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8553658, 0.8553655)
7: (-11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7645159, 0.7645159)
8: (-2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.6016331, 0.6016332)
9: (-3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5812590, 0.5812589)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.33 + 34.37 = 57.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4081061, upper bound: 0.4081070

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4049258
time: 4.28 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4081020
time: 4.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.62 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.62
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4049258
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.62
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4081020

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 8.0396624, 9.2177811, 8.0395517, 9.2213478, -0.5655613, 0.5619678
1: -19.7023907, -17.7050819, -19.7024689, -17.6927280, -0.9176550, 0.9053004
2: -4.7316461, -3.3979287, -4.7370501, -3.3978169, -0.8922272, 0.8977804
3: -11.3078899, -9.8786421, -11.3193445, -9.8785133, -0.6921358, 0.7036636
4: -11.1945438, -9.3358955, -11.1946392, -9.3267460, -0.8265662, 0.8173289
5: -7.2789278, -5.9082370, -7.2789679, -5.9062653, -0.7598057, 0.7578495
6: -4.2404413, -2.8224850, -4.2405648, -2.8082240, -0.8460436, 0.8317273
7: -11.7656307, -10.0610485, -11.7829161, -10.0610094, -0.7375069, 0.7547727
8: -2.8661280, -1.6223488, -2.8680534, -1.6223152, -0.5981071, 0.6001104
9: -3.6998680, -2.3671989, -3.6999886, -2.3637426, -0.5790706, 0.5756402

Time for backsubstitution: 20.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049249, upper bound: 0.4049258
time: 4.59 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049249, upper bound: 0.4049257
time: 5.41 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 8.0241289, 9.2233267, 8.0394897, 9.2233334, -0.5780835, 0.5681825
1: -19.7488518, -17.6847363, -19.7025108, -17.6858616, -0.9356894, 0.9229777
2: -4.7409859, -3.3758552, -4.7400570, -3.3977566, -0.9007034, 0.9108377
3: -11.3264265, -9.8350515, -11.3257065, -9.8784437, -0.7100043, 0.7187985
4: -11.2305746, -9.3210812, -11.1946898, -9.3216543, -0.8391039, 0.8316641
5: -7.2868309, -5.9044747, -7.2789869, -5.9051633, -0.7691131, 0.7602487
6: -4.2952638, -2.8003056, -4.2406330, -2.8002925, -0.8686843, 0.8475437
7: -11.7937355, -9.9971838, -11.7925186, -10.0609827, -0.7541533, 0.7705142
8: -2.8698413, -1.6143429, -2.8691239, -1.6222959, -0.6020105, 0.6088904
9: -3.7138965, -2.3611155, -3.7000561, -2.3618169, -0.5869346, 0.5797373

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6163

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4061506, upper bound: 0.4080992
time: 5.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080997, upper bound: 0.4081003
time: 4.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.39 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.39
Output dim: 0, lower bound: -0.4049249, upper bound: 0.4049258
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 32.39
Output dim: 0, lower bound: -0.4049249, upper bound: 0.4049257
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.39
Output dim: 0, lower bound: -0.4061506, upper bound: 0.4080992
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.39
Output dim: 0, lower bound: -0.4080997, upper bound: 0.4081003

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 8.0247116, 9.2198772, 8.0450630, 9.2164459, -0.5707221, 0.5590820
1: -19.7375793, -17.6860619, -19.6798897, -17.6991882, -0.9047074, 0.8986738
2: -4.7324677, -3.3763356, -4.7227550, -3.4060779, -0.8813362, 0.8920386
3: -11.3195696, -9.8364668, -11.3118944, -9.8879585, -0.6911852, 0.7027377
4: -11.2238865, -9.3212366, -11.1812687, -9.3285370, -0.8222933, 0.8180678
5: -7.2848926, -5.9048114, -7.2750039, -5.9067154, -0.7655621, 0.7555697
6: -4.2948141, -2.8098929, -4.2290568, -2.8194411, -0.8491583, 0.8262212
7: -11.7935953, -10.0018616, -11.7887201, -10.0704775, -0.7446821, 0.7597495
8: -2.8653693, -1.6144652, -2.8600631, -1.6253591, -0.5942163, 0.5994238
9: -3.7134197, -2.3734004, -3.6879306, -2.3864164, -0.5615780, 0.5520945

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6163

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4061505, upper bound: 0.4061498
time: 8.50 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4061505, upper bound: 0.4080993
time: 6.03 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233248, 8.0394897, 9.2233305, -0.5703281, 0.5681818
1: -19.7488480, -17.6847343, -19.7025070, -17.6858635, -0.9247398, 0.8971775
2: -4.7409821, -3.3758543, -4.7400532, -3.3977549, -0.8989825, 0.8933821
3: -11.3264265, -9.8350506, -11.3257046, -9.8784447, -0.7059810, 0.7037499
4: -11.2305737, -9.3210802, -11.1946869, -9.3216562, -0.8325930, 0.8164268
5: -7.2868299, -5.9044752, -7.2789879, -5.9051628, -0.7691123, 0.7595623
6: -4.2952614, -2.8003085, -4.2406311, -2.8002989, -0.8453462, 0.8471494
7: -11.7937374, -9.9971876, -11.7925167, -10.0609884, -0.7455389, 0.7665861
8: -2.8698411, -1.6143432, -2.8691201, -1.6222963, -0.6019225, 0.6019826
9: -3.7138965, -2.3611159, -3.7000561, -2.3618178, -0.5601727, 0.5721141

Time for backsubstitution: 20.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 6163

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4061499
time: 6.19 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4080992
time: 5.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.89 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 32.89
Output dim: 0, lower bound: -0.4061505, upper bound: 0.4061498
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.89
Output dim: 0, lower bound: -0.4061505, upper bound: 0.4080993
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.89
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4061499
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.89
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4080992

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0450630, 9.2164459, -0.5674176, 0.5625329
1: -19.7488499, -17.6847363, -19.6798897, -17.6991882, -0.9051418, 0.8950700
2: -4.7409806, -3.3758547, -4.7227550, -3.4060779, -0.8842592, 0.8856699
3: -11.3264246, -9.8350525, -11.3118944, -9.8879585, -0.6913986, 0.6985748
4: -11.2305698, -9.3210802, -11.1812687, -9.3285370, -0.8224602, 0.8138254
5: -7.2868280, -5.9044747, -7.2750039, -5.9067154, -0.7677510, 0.7557631
6: -4.2952619, -2.8003111, -4.2290568, -2.8194411, -0.8395133, 0.8312354
7: -11.7937374, -9.9971905, -11.7887201, -10.0704775, -0.7443419, 0.7605578
8: -2.8698380, -1.6143413, -2.8600631, -1.6253591, -0.5988016, 0.5966911
9: -3.7138956, -2.3611169, -3.6879306, -2.3864164, -0.5505534, 0.5527651

Time for backsubstitution: 21.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029740, upper bound: 0.4080989
time: 6.47 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029740, upper bound: 0.4081007
time: 4.14 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 8.0297012, 9.2164392, 8.0394897, 9.2233305, -0.5671960, 0.5612915
1: -19.7262287, -17.6980572, -19.7025070, -17.6858635, -0.9019568, 0.8982434
2: -4.7236834, -3.3841784, -4.7400532, -3.3977549, -0.8816392, 0.8882868
3: -11.3126183, -9.8445644, -11.3257046, -9.8784447, -0.6920636, 0.6978981
4: -11.2171507, -9.3279581, -11.1946869, -9.3216562, -0.8191311, 0.8171532
5: -7.2828445, -5.9060254, -7.2789879, -5.9051628, -0.7646286, 0.7589061
6: -4.2836785, -2.8194540, -4.2406311, -2.8002989, -0.8427641, 0.8279853
7: -11.7899418, -10.0066776, -11.7925167, -10.0609884, -0.7493742, 0.7572410
8: -2.8607817, -1.6174045, -2.8691201, -1.6222963, -0.5926292, 0.6009517
9: -3.7017713, -2.3857145, -3.7000561, -2.3618178, -0.5554808, 0.5478379

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4061507
time: 5.25 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4061513
time: 4.02 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0394897, 9.2233305, -0.5715518, 0.5608222
1: -19.7488499, -17.6847363, -19.7025070, -17.6858635, -0.9098616, 0.8971763
2: -4.7409806, -3.3758547, -4.7400532, -3.3977549, -0.8832884, 0.8946545
3: -11.3264246, -9.8350525, -11.3257046, -9.8784447, -0.6949699, 0.7057707
4: -11.2305698, -9.3210802, -11.1946869, -9.3216562, -0.8235445, 0.8164272
5: -7.2868280, -5.9044747, -7.2789879, -5.9051628, -0.7684278, 0.7595623
6: -4.2952619, -2.8003111, -4.2406311, -2.8002989, -0.8468044, 0.8245420
7: -11.7937374, -9.9971905, -11.7925167, -10.0609884, -0.7455389, 0.7618544
8: -2.8698380, -1.6143413, -2.8691201, -1.6222963, -0.5951096, 0.6022056
9: -3.7138956, -2.3611169, -3.7000561, -2.3618178, -0.5632657, 0.5533872

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4080988
time: 7.45 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4081007
time: 4.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.97 seconds
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 0, lower bound: -0.4029740, upper bound: 0.4080989
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 0, lower bound: -0.4029740, upper bound: 0.4081007
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.97
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4061507
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.97
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4061513
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4080988
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.97
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4081007

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0452356, 9.2108965, -0.5608990, 0.5612986
1: -19.7488499, -17.6847363, -19.6797714, -17.7184067, -0.8841622, 0.8923944
2: -4.7409806, -3.3758547, -4.7143421, -3.4062495, -0.8829360, 0.8762317
3: -11.3264246, -9.8350525, -11.2940788, -9.8881550, -0.6883781, 0.6786849
4: -11.2305698, -9.3210802, -11.1811228, -9.3427782, -0.8064985, 0.8113148
5: -7.2868280, -5.9044747, -7.2749424, -5.9097929, -0.7646344, 0.7568979
6: -4.2952619, -2.8003111, -4.2288651, -2.8416333, -0.8161848, 0.8296690
7: -11.7937374, -9.9971905, -11.7618351, -10.0705452, -0.7454712, 0.7335372
8: -2.8698380, -1.6143413, -2.8570662, -1.6254129, -0.5985055, 0.5932686
9: -3.7138956, -2.3611169, -3.6877427, -2.3917966, -0.5449997, 0.5524541

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4069457
time: 4.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4080956
time: 4.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0297012, 9.2164392, -0.5630491, 0.5642921
1: -19.7488499, -17.6847363, -19.7262287, -17.6980572, -0.9031630, 0.8953466
2: -4.7409806, -3.3758547, -4.7236834, -3.3841784, -0.8852739, 0.8835878
3: -11.3264246, -9.8350525, -11.3126183, -9.8445644, -0.6920462, 0.6964977
4: -11.2305698, -9.3210802, -11.2171507, -9.3279581, -0.8218036, 0.8140864
5: -7.2868280, -5.9044747, -7.2828445, -5.9060254, -0.7607338, 0.7575901
6: -4.2952619, -2.8003111, -4.2836785, -2.8194540, -0.8302774, 0.8331375
7: -11.7937374, -9.9971905, -11.7899418, -10.0066776, -0.7468877, 0.7543161
8: -2.8698380, -1.6143413, -2.8607817, -1.6174045, -0.5996723, 0.5935013
9: -3.7138956, -2.3611169, -3.7017713, -2.3857145, -0.5509765, 0.5534924

Time for backsubstitution: 22.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4069459
time: 6.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4080963
time: 4.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0396614, 9.2177792, -0.5647342, 0.5595884
1: -19.7488499, -17.6847363, -19.7023907, -17.7050800, -0.8888838, 0.8987012
2: -4.7409806, -3.3758547, -4.7316422, -3.3979292, -0.8833570, 0.8849998
3: -11.3264246, -9.8350525, -11.3078871, -9.8786440, -0.6942157, 0.6851815
4: -11.2305698, -9.3210802, -11.1945391, -9.3358965, -0.8075831, 0.8159845
5: -7.2868280, -5.9044747, -7.2789249, -5.9082384, -0.7653155, 0.7606971
6: -4.2952619, -2.8003111, -4.2404399, -2.8224905, -0.8231785, 0.8302879
7: -11.7937374, -9.9971905, -11.7656307, -10.0610571, -0.7513649, 0.7348342
8: -2.8698380, -1.6143413, -2.8661261, -1.6223495, -0.5948137, 0.5986912
9: -3.7138956, -2.3611169, -3.6998680, -2.3671989, -0.5575261, 0.5549288

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069457
time: 4.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080956
time: 4.08 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0241299, 9.2233238, -0.5625799, 0.5625799
1: -19.7488499, -17.6847363, -19.7488499, -17.6847363, -0.8974578, 0.8974581
2: -4.7409806, -3.3758547, -4.7409806, -3.3758547, -0.8844118, 0.8844118
3: -11.3264246, -9.8350525, -11.3264246, -9.8350525, -0.6956213, 0.6956213
4: -11.2305698, -9.3210802, -11.2305698, -9.3210802, -0.8167281, 0.8167281
5: -7.2868280, -5.9044747, -7.2868280, -5.9044747, -0.7613893, 0.7613895
6: -4.2952619, -2.8003111, -4.2952619, -2.8003111, -0.8264284, 0.8264282
7: -11.7937374, -9.9971905, -11.7937374, -9.9971905, -0.7499781, 0.7499781
8: -2.8698380, -1.6143413, -2.8698380, -1.6143413, -0.5959826, 0.5959826
9: -3.7138956, -2.3611169, -3.7138956, -2.3611169, -0.5545466, 0.5545467

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069448
time: 6.93 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080947
time: 7.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 35.87 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4069457
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4080956
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4069459
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4080963
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069457
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080956
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069448
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 35.87
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080947

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 8.0128479, 9.2241163, 8.0452366, 9.2108917, -0.5651882, 0.5627735
1: -19.7529221, -17.6835136, -19.6797714, -17.7184067, -0.8844295, 0.8934432
2: -4.7433681, -3.3707721, -4.7143412, -3.4062502, -0.8835616, 0.8784246
3: -11.3343868, -9.8338652, -11.2940779, -9.8881569, -0.6899865, 0.6794875
4: -11.2308111, -9.3036242, -11.1811209, -9.3427753, -0.8070540, 0.8126738
5: -7.2962832, -5.9024706, -7.2749414, -5.9097986, -0.7684669, 0.7584937
6: -4.2980685, -2.7965381, -4.2288618, -2.8416338, -0.8192759, 0.8308156
7: -11.8269386, -9.9965382, -11.7618361, -10.0705557, -0.7525663, 0.7346689
8: -2.8744562, -1.6128020, -2.8570669, -1.6254146, -0.6017696, 0.5949728
9: -3.7157774, -2.3609035, -3.6877432, -2.3917966, -0.5469555, 0.5520121

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080944
time: 5.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080949
time: 4.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 8.0128479, 9.2241163, 8.0296984, 9.2164364, -0.5722872, 0.5660233
1: -19.7529221, -17.6835136, -19.7262287, -17.6980591, -0.9034808, 0.8963962
2: -4.7433681, -3.3707721, -4.7236805, -3.3841774, -0.8859012, 0.8889868
3: -11.3343868, -9.8338652, -11.3126173, -9.8445654, -0.6936529, 0.6972761
4: -11.2308111, -9.3036242, -11.2171450, -9.3279581, -0.8219872, 0.8154444
5: -7.2962832, -5.9024706, -7.2828436, -5.9060287, -0.7712202, 0.7591860
6: -4.2980685, -2.7965381, -4.2836752, -2.8194523, -0.8332925, 0.8342810
7: -11.8269386, -9.9965382, -11.7899408, -10.0066872, -0.7539904, 0.7554293
8: -2.8744562, -1.6128020, -2.8607793, -1.6174057, -0.6031899, 0.5952227
9: -3.7157774, -2.3609035, -3.7017715, -2.3857145, -0.5529323, 0.5530503

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080966
time: 4.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080951
time: 8.10 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 8.0128479, 9.2241163, 8.0396633, 9.2177763, -0.5693895, 0.5618119
1: -19.7529221, -17.6835136, -19.7023869, -17.7050819, -0.8891513, 0.8997769
2: -4.7433681, -3.3707721, -4.7316370, -3.3979299, -0.8839574, 0.8883834
3: -11.3343868, -9.8338652, -11.3078880, -9.8786478, -0.6958247, 0.6860824
4: -11.2308111, -9.3036242, -11.1945324, -9.3358974, -0.8081424, 0.8195281
5: -7.2962832, -5.9024706, -7.2789249, -5.9082403, -0.7704334, 0.7622926
6: -4.2980685, -2.7965381, -4.2404370, -2.8224900, -0.8265951, 0.8330760
7: -11.8269386, -9.9965382, -11.7656307, -10.0610666, -0.7610061, 0.7359661
8: -2.8744562, -1.6128020, -2.8661242, -1.6223507, -0.6028005, 0.6008840
9: -3.7157774, -2.3609035, -3.6998684, -2.3671994, -0.5601331, 0.5544832

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080959
time: 4.46 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080959
time: 5.07 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 8.0128479, 9.2241163, 8.0241308, 9.2233181, -0.5736014, 0.5648018
1: -19.7529221, -17.6835136, -19.7488480, -17.6847382, -0.9014285, 0.8985369
2: -4.7433681, -3.3707721, -4.7409782, -3.3758554, -0.8850274, 0.8952451
3: -11.3343868, -9.8338652, -11.3264236, -9.8350554, -0.6995015, 0.6964188
4: -11.2308111, -9.3036242, -11.2305641, -9.3210793, -0.8172765, 0.8254043
5: -7.2962832, -5.9024706, -7.2868271, -5.9044785, -0.7718763, 0.7629850
6: -4.2980685, -2.7965381, -4.2952604, -2.8003104, -0.8294435, 0.8323848
7: -11.8269386, -9.9965382, -11.7937355, -9.9971981, -0.7643478, 0.7510910
8: -2.8744562, -1.6128020, -2.8698370, -1.6143446, -0.6042209, 0.5977075
9: -3.7157774, -2.3609035, -3.7138934, -2.3611178, -0.5614860, 0.5541019

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 110

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080966
time: 4.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080952
time: 6.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 33.05 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080944
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080949
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080966
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080951
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080959
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080959
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080966
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 33.05
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080952

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 8.0129948, 9.2241163, 8.0458288, 9.2095327, -0.5634301, 0.5606728
1: -19.7529240, -17.6835442, -19.6797600, -17.7191257, -0.8836329, 0.8932470
2: -4.7433434, -3.3709974, -4.7133532, -3.4067385, -0.8825207, 0.8754330
3: -11.3342991, -9.8338881, -11.2935810, -9.8897772, -0.6880803, 0.6788584
4: -11.2308064, -9.3037891, -11.1782951, -9.3429947, -0.8064005, 0.8097248
5: -7.2960424, -5.9024839, -7.2745028, -5.9114981, -0.7664981, 0.7582035
6: -4.2980042, -2.7966974, -4.2276802, -2.8418584, -0.8185802, 0.8283634
7: -11.8261032, -9.9965401, -11.7614813, -10.0759964, -0.7462437, 0.7342172
8: -2.8743119, -1.6128368, -2.8565593, -1.6262195, -0.5995471, 0.5940068
9: -3.7156394, -2.3609073, -3.6873164, -2.3922341, -0.5448159, 0.5512685

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4582

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4015854, upper bound: 0.4080940
time: 4.41 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018290, upper bound: 0.4080930
time: 7.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 8.0128479, 9.2241163, 8.0339241, 9.2116890, -0.5665674, 0.5671200
1: -19.7529221, -17.6835136, -19.6838417, -17.7171745, -0.8854711, 0.8935913
2: -4.7433681, -3.3707721, -4.7167292, -3.4011323, -0.8864865, 0.8796570
3: -11.3343868, -9.8338652, -11.3020430, -9.8869619, -0.6908334, 0.6812004
4: -11.2308111, -9.3036242, -11.1813593, -9.3253222, -0.8080294, 0.8132286
5: -7.2962832, -5.9024706, -7.2844110, -5.9077659, -0.7700682, 0.7666802
6: -4.2980685, -2.7965381, -4.2316885, -2.8378589, -0.8207397, 0.8341711
7: -11.8269386, -9.9965382, -11.7950411, -10.0698977, -0.7536960, 0.7412974
8: -2.8744562, -1.6128020, -2.8616891, -1.6238639, -0.6039102, 0.5981658
9: -3.7157774, -2.3609035, -3.6896448, -2.3915815, -0.5470264, 0.5546330

Time for backsubstitution: 22.11 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.70 + 556.09 = 613.79 seconds
