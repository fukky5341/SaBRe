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
execution time: IAR + RelationalAnalysis = 23.73 + 35.01 = 58.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4081061, upper bound: 0.4081070

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4049258
time: 4.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4081020
time: 4.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.96 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.96
Output dim: 0, lower bound: -0.4081015, upper bound: 0.4049258
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.96
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

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 6163

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4029738
time: 5.36 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4049240
time: 4.09 seconds

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

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6163
type: B, layer: 1, pos: 6163
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6163

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4061498
time: 5.72 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4080994
time: 4.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.57 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 32.57
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4029738
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 32.57
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4049240
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 32.57
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4061498
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 32.57
Output dim: 0, lower bound: -0.4080998, upper bound: 0.4080994

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 8.0452356, 9.2108965, 8.0401335, 9.2178993, -0.5564607, 0.5546510
1: -19.6797714, -17.7184067, -19.6911945, -17.6940556, -0.8933551, 0.8797340
2: -4.7143421, -3.4062495, -4.7285318, -3.3982978, -0.8734617, 0.8784184
3: -11.2940788, -9.8881550, -11.3124886, -9.8799295, -0.6760910, 0.6858919
4: -11.1811228, -9.3427782, -11.1879530, -9.3269062, -0.8129692, 0.8037555
5: -7.2749424, -5.9097929, -7.2770305, -5.9066057, -0.7551272, 0.7542982
6: -4.2288651, -2.8416333, -4.2401094, -2.8178103, -0.8247194, 0.8122311
7: -11.7618351, -10.0705452, -11.7827721, -10.0656834, -0.7285347, 0.7453005
8: -2.8570662, -1.6254129, -2.8635826, -1.6224377, -0.5886596, 0.5923165
9: -3.6877427, -2.3917966, -3.6995125, -2.3760252, -0.5514300, 0.5502113

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4029738
time: 6.54 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4029737
time: 5.99 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 8.0396614, 9.2177792, 8.0395508, 9.2213459, -0.5655613, 0.5546081
1: -19.7023907, -17.7050800, -19.7024651, -17.6927280, -0.8918531, 0.9041795
2: -4.7316422, -3.3979292, -4.7370496, -3.3978167, -0.8748116, 0.8968527
3: -11.3078871, -9.8786440, -11.3193445, -9.8785143, -0.6771035, 0.7027074
4: -11.1945391, -9.3358965, -11.1946373, -9.3267460, -0.8113294, 0.8169093
5: -7.2789249, -5.9082384, -7.2789650, -5.9062653, -0.7591200, 0.7578492
6: -4.2404399, -2.8224905, -4.2405653, -2.8082263, -0.8460402, 0.8087261
7: -11.7656307, -10.0610571, -11.7829132, -10.0610104, -0.7370310, 0.7461565
8: -2.8661261, -1.6223495, -2.8680511, -1.6223142, -0.5912056, 0.6000230
9: -3.6998680, -2.3671989, -3.6999917, -2.3637435, -0.5714493, 0.5492910

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4049229
time: 6.70 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4049240
time: 4.21 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 8.0297012, 9.2164392, 8.0400724, 9.2198839, -0.5671957, 0.5608659
1: -19.7262287, -17.6980572, -19.6912346, -17.6871872, -0.9113836, 0.8974004
2: -4.7236834, -3.3841784, -4.7315388, -3.3982377, -0.8819270, 0.8875816
3: -11.3126183, -9.8445644, -11.3188486, -9.8798580, -0.6939609, 0.6976893
4: -11.2171507, -9.3279581, -11.1880074, -9.3218136, -0.8254972, 0.8169864
5: -7.2828445, -5.9060254, -7.2770529, -5.9055033, -0.7644358, 0.7566962
6: -4.2836785, -2.8194540, -4.2401757, -2.8098812, -0.8426421, 0.8280478
7: -11.7899418, -10.0066776, -11.7923717, -10.0656605, -0.7451828, 0.7610329
8: -2.8607817, -1.6174045, -2.8646512, -1.6224179, -0.5925626, 0.5995344
9: -3.7017713, -2.3857145, -3.6995790, -2.3741012, -0.5548109, 0.5543084

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4061508
time: 4.63 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4061515
time: 3.89 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0394907, 9.2233315, -0.5743086, 0.5608227
1: -19.7488499, -17.6847363, -19.7025108, -17.6858616, -0.9098628, 0.9178529
2: -4.7409806, -3.3758547, -4.7400532, -3.3977556, -0.8832893, 0.9030070
3: -11.3264246, -9.8350525, -11.3257046, -9.8784447, -0.6949718, 0.7124913
4: -11.2305698, -9.3210802, -11.1946907, -9.3216562, -0.8236995, 0.8272851
5: -7.2868280, -5.9044747, -7.2789869, -5.9051623, -0.7684283, 0.7602468
6: -4.2952619, -2.8003111, -4.2406330, -2.8002956, -0.8586776, 0.8245416
7: -11.7937374, -9.9971905, -11.7925167, -10.0609827, -0.7536769, 0.7618544
8: -2.8698380, -1.6143413, -2.8691206, -1.6222959, -0.5951098, 0.6060139
9: -3.7138956, -2.3611169, -3.7000575, -2.3618183, -0.5748293, 0.5533882

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4080990
time: 6.50 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4081004
time: 5.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.19 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4029738
NS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4029737
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4049229
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049237, upper bound: 0.4049240
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4061508
NS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4061515
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4080990
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 34.19
Output dim: 0, lower bound: -0.4049232, upper bound: 0.4081004

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0396614, 9.2177820, -0.5677900, 0.5595888
1: -19.7488499, -17.6847363, -19.7023926, -17.7050781, -0.8888848, 0.9151773
2: -4.7409806, -3.3758547, -4.7316432, -3.3979285, -0.8833580, 0.8935678
3: -11.3264246, -9.8350525, -11.3078880, -9.8786430, -0.6942168, 0.6926011
4: -11.2305698, -9.3210802, -11.1945419, -9.3358965, -0.8077376, 0.8247744
5: -7.2868280, -5.9044747, -7.2789278, -5.9082365, -0.7653160, 0.7613819
6: -4.2952619, -2.8003111, -4.2404408, -2.8224864, -0.8353486, 0.8302886
7: -11.7937374, -9.9971905, -11.7656317, -10.0610523, -0.7548177, 0.7348344
8: -2.8698380, -1.6143413, -2.8661270, -1.6223497, -0.5948137, 0.6025909
9: -3.7138956, -2.3611169, -3.6998687, -2.3671997, -0.5692755, 0.5549299

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6163

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4080989
time: 5.93 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4080989
time: 7.84 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0241299, 9.2233248, -0.5699394, 0.5625800
1: -19.7488499, -17.6847363, -19.7488480, -17.6847343, -0.8974586, 0.9181293
2: -4.7409806, -3.3758547, -4.7409821, -3.3758543, -0.8844137, 0.9009008
3: -11.3264246, -9.8350525, -11.3264265, -9.8350506, -0.6956227, 0.7104101
4: -11.2305698, -9.3210802, -11.2305737, -9.3210802, -0.8167281, 0.8275478
5: -7.2868280, -5.9044747, -7.2868299, -5.9044752, -0.7613893, 0.7620742
6: -4.2952619, -2.8003111, -4.2952614, -2.8003085, -0.8494267, 0.8264287
7: -11.7937374, -9.9971905, -11.7937374, -9.9971876, -0.7562327, 0.7499783
8: -2.8698380, -1.6143413, -2.8698411, -1.6143432, -0.5959823, 0.6027946
9: -3.7138956, -2.3611169, -3.7138965, -2.3611159, -0.5752528, 0.5545478

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6163
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6163

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4081003
time: 4.52 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4081006
time: 3.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.68 seconds
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4080989
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4080989
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4081003
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 0, lower bound: -0.4029739, upper bound: 0.4081006

## BFS NS instance: NS_A2_A2_B1_B1

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

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018305, upper bound: 0.4080956
time: 4.49 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029694, upper bound: 0.4080947
time: 6.63 seconds

## BFS NS instance: NS_A2_A2_B1_B2

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

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018305, upper bound: 0.4080946
time: 5.82 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029694, upper bound: 0.4080948
time: 6.88 seconds

## BFS NS instance: NS_A2_A2_B2_B1

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

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018305, upper bound: 0.4080962
time: 3.86 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029694, upper bound: 0.4080962
time: 4.21 seconds

## BFS NS instance: NS_A2_A2_B2_B2

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

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069463
time: 3.99 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080962
time: 4.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 30.73 seconds
NS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4018305, upper bound: 0.4080956
NS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4029694, upper bound: 0.4080947
NS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4018305, upper bound: 0.4080946
NS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4029694, upper bound: 0.4080948
NS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4018305, upper bound: 0.4080962
NS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4029694, upper bound: 0.4080962
NS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069463
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.73
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080962

## BFS NS instance: NS_A2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0458288, 9.2095327, -0.5593133, 0.5601864
1: -19.7488499, -17.6847363, -19.6797600, -17.7191257, -0.8833749, 0.8923426
2: -4.7409806, -3.3758547, -4.7133532, -3.4067385, -0.8819089, 0.8744233
3: -11.3264246, -9.8350525, -11.2935810, -9.8897772, -0.6866555, 0.6780741
4: -11.2305698, -9.3210802, -11.1782951, -9.3429947, -0.8062172, 0.8084767
5: -7.2868280, -5.9044747, -7.2745028, -5.9114981, -0.7629251, 0.7566955
6: -4.2952619, -2.8003111, -4.2276802, -2.8418584, -0.8155859, 0.8277276
7: -11.7937374, -9.9971905, -11.7614813, -10.0759964, -0.7399080, 0.7334545
8: -2.8698380, -1.6143413, -2.8565593, -1.6262195, -0.5970709, 0.5923600
9: -3.7138956, -2.3611169, -3.6873164, -2.3922341, -0.5438554, 0.5517147

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4069442
time: 4.98 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018304, upper bound: 0.4080944
time: 4.84 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 8.0241308, 9.2233181, 8.0339241, 9.2116890, -0.5619053, 0.5660646
1: -19.7488480, -17.6847382, -19.6838417, -17.7171745, -0.8852036, 0.8926609
2: -4.7409782, -3.3758554, -4.7167292, -3.4011323, -0.8850803, 0.8768556
3: -11.3264236, -9.8350554, -11.3020430, -9.8869619, -0.6891536, 0.6803074
4: -11.2305641, -9.3210793, -11.1813593, -9.3253222, -0.8078439, 0.8118687
5: -7.2868271, -5.9044785, -7.2844110, -5.9077659, -0.7662339, 0.7673917
6: -4.2952604, -2.8003104, -4.2316885, -2.8378589, -0.8173149, 0.8327651
7: -11.7937355, -9.9971981, -11.7950411, -10.0698977, -0.7466009, 0.7405330
8: -2.8698370, -1.6143446, -2.8616891, -1.6238639, -0.6002092, 0.5959153
9: -3.7138934, -2.3611178, -3.6896448, -2.3915815, -0.5445569, 0.5544161

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4069451
time: 4.00 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029698, upper bound: 0.4080953
time: 4.22 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0402622, 9.2164154, -0.5631486, 0.5584749
1: -19.7488499, -17.6847363, -19.7023773, -17.7057991, -0.8880975, 0.8986452
2: -4.7409806, -3.3758547, -4.7306547, -3.3984177, -0.8823643, 0.8831925
3: -11.3264246, -9.8350525, -11.3073921, -9.8802662, -0.6924934, 0.6845713
4: -11.2305698, -9.3210802, -11.1917152, -9.3361130, -0.8073015, 0.8131490
5: -7.2868280, -5.9044747, -7.2784872, -5.9099426, -0.7636096, 0.7604942
6: -4.2952619, -2.8003111, -4.2392521, -2.8227153, -0.8225796, 0.8283482
7: -11.7937374, -9.9971905, -11.7652769, -10.0665073, -0.7458017, 0.7347512
8: -2.8698380, -1.6143413, -2.8656154, -1.6231570, -0.5933793, 0.5977829
9: -3.7138956, -2.3611169, -3.6994395, -2.3676360, -0.5563818, 0.5541875

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4069454
time: 4.39 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080959
time: 4.62 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 8.0241308, 9.2233181, 8.0283794, 9.2185707, -0.5656087, 0.5688084
1: -19.7488480, -17.6847382, -19.7064648, -17.7038536, -0.8899391, 0.9020778
2: -4.7409782, -3.3758554, -4.7340422, -3.3928103, -0.8901794, 0.8864236
3: -11.3264236, -9.8350554, -11.3158522, -9.8774595, -0.6950297, 0.6878980
4: -11.2305641, -9.3210793, -11.1947775, -9.3184443, -0.8089283, 0.8165348
5: -7.2868271, -5.9044785, -7.2883987, -5.9062247, -0.7669113, 0.7711954
6: -4.2952604, -2.8003104, -4.2432647, -2.8187165, -0.8249888, 0.8333099
7: -11.7937355, -9.9971981, -11.7988358, -10.0604105, -0.7521281, 0.7418346
8: -2.8698370, -1.6143446, -2.8707480, -1.6208038, -0.5965194, 0.6021914
9: -3.7138934, -2.3611178, -3.7017689, -2.3669851, -0.5577406, 0.5591080

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 4657
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of NS_A2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4069445
time: 6.46 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4029697, upper bound: 0.4080947
time: 6.59 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: 8.0241299, 9.2233238, 8.0302982, 9.2150774, -0.5614666, 0.5631764
1: -19.7488499, -17.6847363, -19.7262192, -17.6987782, -0.9023750, 0.8952951
2: -4.7409806, -3.3758547, -4.7226949, -3.3846693, -0.8842511, 0.8817840
3: -11.3264246, -9.8350525, -11.3121233, -9.8461838, -0.6903231, 0.6958792
4: -11.2305698, -9.3210802, -11.2143240, -9.3281746, -0.8215187, 0.8112483
5: -7.2868280, -5.9044747, -7.2824001, -5.9077325, -0.7590277, 0.7573786
6: -4.2952619, -2.8003111, -4.2824917, -2.8196762, -0.8297162, 0.8311944
7: -11.7937374, -9.9971905, -11.7895947, -10.0121307, -0.7413254, 0.7542481
8: -2.8698380, -1.6143413, -2.8602738, -1.6182127, -0.5982428, 0.5925909
9: -3.7138956, -2.3611169, -3.7013428, -2.3861492, -0.5498322, 0.5527287

Time for backsubstitution: 22.26 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.74 + 556.45 = 615.18 seconds
