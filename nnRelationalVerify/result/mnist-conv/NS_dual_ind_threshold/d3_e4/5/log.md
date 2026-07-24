## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.3532293525


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5333910, 2.5333900)
1: (-12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5630426, 2.5630426)
2: (-13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5471597, 2.5471601)
3: (-9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7141438, 2.7141433)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5921633, 1.5921636)
5: (-11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5612335, 2.5612335)
6: (-17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8651266, 2.8651257)
7: (-6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1917086, 2.1917083)
8: (-2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7734652, 1.7734652)
9: (2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2999578, 2.2999575)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.89 + 33.88 = 56.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.3600280, upper bound: 1.3600274

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597460, upper bound: 1.3509754
time: 4.99 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597460, upper bound: 1.3597474
time: 4.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.34 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.34
Output dim: 9, lower bound: -1.3597460, upper bound: 1.3509754
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.34
Output dim: 9, lower bound: -1.3597460, upper bound: 1.3597474

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -14.2953243, -10.2914839, -14.2979259, -10.2878437, -2.5280714, 2.5276518
1: -12.4783468, -8.9430885, -12.4861097, -8.9376755, -2.5456753, 2.5480337
2: -13.4055119, -10.1823168, -13.4084225, -10.1802893, -2.5397305, 2.5419388
3: -9.8860092, -6.9041572, -9.8885040, -6.9029617, -2.7091708, 2.7057753
4: -4.5463209, -2.4078248, -4.5532451, -2.4014208, -1.5764737, 1.5774460
5: -11.0696678, -7.3708878, -11.0722752, -7.3679771, -2.5535975, 2.5553637
6: -17.5701237, -13.6092281, -17.5752354, -13.6046534, -2.8529119, 2.8541684
7: -6.4289103, -3.6039596, -6.4322462, -3.5995250, -2.1809998, 2.1818237
8: -2.0346713, 0.1787167, -2.0386643, 0.1811047, -1.7652001, 1.7667551
9: 2.4283915, 5.1467171, 2.4199867, 5.1529918, -2.2814734, 2.2835472

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3495659
time: 5.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3509735
time: 4.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -14.3000641, -10.2871685, -14.3000641, -10.2871685, -2.5330467, 2.5331035
1: -12.4945688, -8.9361629, -12.4945755, -8.9361629, -2.5570974, 2.5630388
2: -13.4097586, -10.1796150, -13.4097595, -10.1796112, -2.5461431, 2.5456119
3: -9.8902359, -6.9025412, -9.8902378, -6.9025407, -2.7136312, 2.7194471
4: -4.5608358, -2.3998001, -4.5608387, -2.3997998, -1.5798414, 1.5921597
5: -11.0733891, -7.3661032, -11.0733929, -7.3660994, -2.5630207, 2.5610595
6: -17.5802135, -13.6031475, -17.5802155, -13.6031466, -2.8609438, 2.8651223
7: -6.4332151, -3.5954461, -6.4332156, -3.5954432, -2.1913681, 2.1832888
8: -2.0399032, 0.1837773, -2.0399032, 0.1837792, -1.7734623, 1.7701535
9: 2.4171581, 5.1602273, 2.4171572, 5.1602297, -2.2999537, 2.2939138

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3583383
time: 4.40 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3597458
time: 4.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.45 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3495659
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3509735
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3583383
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 9, lower bound: -1.3597446, upper bound: 1.3597458

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -14.2893963, -10.2919312, -14.2872219, -10.2886572, -2.5196347, 2.5152993
1: -12.4757147, -8.9453640, -12.4813251, -8.9417686, -2.5395732, 2.5406790
2: -13.3989162, -10.1862259, -13.3964109, -10.1877174, -2.5227585, 2.5201917
3: -9.8842354, -6.9174547, -9.8853855, -6.9267368, -2.6875563, 2.6915274
4: -4.5455298, -2.4095378, -4.5517306, -2.4045589, -1.5661695, 1.5674248
5: -11.0668669, -7.3804274, -11.0669603, -7.3852563, -2.5288763, 2.5377846
6: -17.5637550, -13.6098785, -17.5633545, -13.6058359, -2.8422260, 2.8388953
7: -6.4216709, -3.6067820, -6.4190893, -3.6047425, -2.1696625, 2.1687369
8: -2.0327244, 0.1750832, -2.0351768, 0.1744943, -1.7523618, 1.7568531
9: 2.4315796, 5.1462569, 2.4257731, 5.1521344, -2.2760301, 2.2760308

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574716, upper bound: 1.3482326
time: 4.76 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3495621
time: 5.34 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -14.2953148, -10.2914820, -14.3089046, -10.2693863, -2.5547543, 2.5400782
1: -12.4783363, -8.9430904, -12.4885473, -8.9046965, -2.5880804, 2.5578346
2: -13.4055080, -10.1823177, -13.4103909, -10.1162567, -2.5867310, 2.5483475
3: -9.8860073, -6.9041777, -9.9418163, -6.8913250, -2.7228012, 2.7581520
4: -4.5463200, -2.4078269, -4.5607710, -2.3914263, -1.5824184, 1.5915089
5: -11.0696640, -7.3709111, -11.1351318, -7.3632240, -2.5537314, 2.6165981
6: -17.5701199, -13.6092310, -17.6113129, -13.5984249, -2.8856850, 2.9095528
7: -6.4288888, -3.6039624, -6.4414272, -3.5604458, -2.2246950, 2.1948640
8: -2.0346680, 0.1787062, -2.0624723, 0.1830802, -1.7768369, 1.7942152
9: 2.4283957, 5.1467156, 2.4121542, 5.1582088, -2.2872114, 2.2913339

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574716, upper bound: 1.3496332
time: 5.01 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3509724
time: 4.35 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -14.2941294, -10.2876158, -14.2893562, -10.2879810, -2.5246153, 2.5207405
1: -12.4919386, -8.9384375, -12.4897919, -8.9402523, -2.5509968, 2.5556855
2: -13.4031563, -10.1835251, -13.3977480, -10.1870413, -2.5291662, 2.5238652
3: -9.8884640, -6.9158363, -9.8871193, -6.9263163, -2.6920156, 2.7051992
4: -4.5600443, -2.4015160, -4.5593238, -2.4029374, -1.5695367, 1.5821402
5: -11.0705881, -7.3756347, -11.0680790, -7.3833742, -2.5383024, 2.5434847
6: -17.5738354, -13.6037979, -17.5683308, -13.6043301, -2.8502712, 2.8498507
7: -6.4259753, -3.5982676, -6.4200583, -3.6006594, -2.1800318, 2.1702008
8: -2.0379591, 0.1801434, -2.0364194, 0.1771684, -1.7606235, 1.7602508
9: 2.4203434, 5.1597672, 2.4229436, 5.1593714, -2.2945111, 2.2863975

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574739, upper bound: 1.3570054
time: 4.47 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3583338
time: 4.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -14.3000565, -10.2871685, -14.3110466, -10.2687073, -2.5597339, 2.5455475
1: -12.4945621, -8.9361629, -12.4970150, -8.9031754, -2.5994992, 2.5728397
2: -13.4097528, -10.1796188, -13.4117241, -10.1156006, -2.5953302, 2.5520186
3: -9.8902330, -6.9025598, -9.9435482, -6.8909044, -2.7272606, 2.7661371
4: -4.5608363, -2.3998032, -4.5683627, -2.3898096, -1.5857847, 1.6062233
5: -11.0733871, -7.3661208, -11.1362438, -7.3613491, -2.5631552, 2.6217096
6: -17.5802078, -13.6031494, -17.6162872, -13.5969200, -2.8937263, 2.9202712
7: -6.4331937, -3.5954485, -6.4423933, -3.5563617, -2.2330141, 2.1963363
8: -2.0398998, 0.1837683, -2.0637150, 0.1857533, -1.7851000, 1.7974131
9: 2.4171619, 5.1602259, 2.4093223, 5.1654453, -2.3056922, 2.3016980

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574716, upper bound: 1.3584068
time: 4.17 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3597419
time: 4.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.60 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3574716, upper bound: 1.3482326
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3495621
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3574716, upper bound: 1.3496332
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3509724
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3574739, upper bound: 1.3570054
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3583338
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3574716, upper bound: 1.3584068
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.60
Output dim: 9, lower bound: -1.3597404, upper bound: 1.3597419

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14.2800293, -10.3245754, -14.2817574, -10.3078213, -2.4925928, 2.4776883
1: -12.4714394, -8.9589453, -12.4788408, -8.9498615, -2.5230856, 2.5209770
2: -13.3864975, -10.1920338, -13.3891411, -10.1910229, -2.5024853, 2.5024467
3: -9.8741417, -6.9207358, -9.8794432, -6.9286480, -2.6742640, 2.6808901
4: -4.5449238, -2.4245219, -4.5513935, -2.4133658, -1.5499661, 1.5463619
5: -11.0581093, -7.3824072, -11.0618076, -7.3864121, -2.5104413, 2.5214462
6: -17.5575600, -13.6354122, -17.5598240, -13.6208220, -2.8153028, 2.8044786
7: -6.4186945, -3.6120958, -6.4173298, -3.6078839, -2.1594343, 2.1579466
8: -2.0269084, 0.1624937, -2.0317450, 0.1670871, -1.7373395, 1.7390440
9: 2.4411793, 5.1450615, 2.4314327, 5.1514344, -2.2623320, 2.2647169

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3482328
time: 4.49 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3482327
time: 4.94 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14.3333693, -10.2892113, -14.2872143, -10.2886772, -2.5582924, 2.5037231
1: -12.4863548, -8.9424267, -12.4813251, -8.9417830, -2.5474777, 2.5352983
2: -13.4027281, -10.1735001, -13.3964005, -10.1877213, -2.5214691, 2.5311909
3: -9.8888168, -6.9061794, -9.8853760, -6.9267387, -2.6913137, 2.7017174
4: -4.5575218, -2.4053645, -4.5517306, -2.4045763, -1.5729928, 1.5651388
5: -11.0715113, -7.3747234, -11.0669556, -7.3852544, -2.5337481, 2.5364380
6: -17.6050568, -13.6085625, -17.5633507, -13.6058474, -2.8795819, 2.8250694
7: -6.4266610, -3.6030998, -6.4190874, -3.6047444, -2.1746473, 2.1722016
8: -2.0522833, 0.1782985, -2.0351715, 0.1744881, -1.7702684, 1.7575080
9: 2.4277053, 5.1499014, 2.4257803, 5.1521335, -2.2770290, 2.2756753

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3495646
time: 5.01 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3495653
time: 4.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14.2859554, -10.3241253, -14.3034477, -10.2885542, -2.5275068, 2.5023026
1: -12.4740601, -8.9566870, -12.4860764, -8.9129429, -2.5715637, 2.5379238
2: -13.3930979, -10.1882954, -13.4031219, -10.1197166, -2.5661931, 2.5305462
3: -9.8759260, -6.9074464, -9.9358501, -6.8930230, -2.7094402, 2.7469425
4: -4.5456843, -2.4228196, -4.5603929, -2.4003322, -1.5661397, 1.5704298
5: -11.0608969, -7.3728914, -11.1298122, -7.3643799, -2.5351439, 2.5996115
6: -17.5639400, -13.6347542, -17.6075325, -13.6134062, -2.8589959, 2.8748693
7: -6.4259157, -3.6092834, -6.4397001, -3.5636911, -2.2144959, 2.1840274
8: -2.0288186, 0.1661229, -2.0587997, 0.1756716, -1.7617836, 1.7762625
9: 2.4379854, 5.1455178, 2.4178181, 5.1574655, -2.2734818, 2.2800262

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3496336
time: 4.54 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3496332
time: 5.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.3393250, -10.2887726, -14.3088980, -10.2694016, -2.5731397, 2.5284481
1: -12.4889994, -8.9402256, -12.4885445, -8.9047089, -2.5960426, 2.5522885
2: -13.4093218, -10.1695557, -13.4103794, -10.1162586, -2.5853200, 2.5595655
3: -9.8906202, -6.8927374, -9.9418087, -6.8913269, -2.7265778, 2.7618265
4: -4.5583153, -2.4036875, -4.5607691, -2.3914437, -1.5892878, 1.5876666
5: -11.0742006, -7.3651800, -11.1351242, -7.3632250, -2.5586667, 2.6128228
6: -17.6116142, -13.6079130, -17.6113091, -13.5984383, -2.9099789, 2.8957489
7: -6.4338665, -3.6003222, -6.4414268, -3.5604501, -2.2297773, 2.1982551
8: -2.0542908, 0.1819239, -2.0624685, 0.1830730, -1.7869244, 1.7948070
9: 2.4245214, 5.1503649, 2.4121618, 5.1582079, -2.2882140, 2.2910047

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3509716
time: 4.24 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3509694
time: 5.94 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -14.2846775, -10.3202534, -14.2838421, -10.3071442, -2.4975872, 2.4830718
1: -12.4876547, -8.9520664, -12.4873018, -8.9483461, -2.5345144, 2.5359998
2: -13.3908319, -10.1893368, -13.3904715, -10.1903477, -2.5088925, 2.5061173
3: -9.8783684, -6.9191179, -9.8811779, -6.9282255, -2.6787224, 2.6945615
4: -4.5594416, -2.4164894, -4.5589871, -2.4117446, -1.5533385, 1.5610895
5: -11.0618382, -7.3776178, -11.0629244, -7.3845367, -2.5198727, 2.5271454
6: -17.5676365, -13.6293287, -17.5647926, -13.6193104, -2.8233495, 2.8154397
7: -6.4230013, -3.6036158, -6.4183006, -3.6038094, -2.1698055, 2.1593714
8: -2.0321622, 0.1675563, -2.0329924, 0.1697617, -1.7456193, 1.7424479
9: 2.4299707, 5.1585712, 2.4286094, 5.1586723, -2.2808428, 2.2750928

Time for backsubstitution: 22.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3570063
time: 5.67 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3570042
time: 4.33 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -14.3380985, -10.2849026, -14.2893467, -10.2879992, -2.5634451, 2.5091572
1: -12.5025940, -8.9354849, -12.4897881, -8.9402599, -2.5589166, 2.5503702
2: -13.4070215, -10.1707897, -13.3977375, -10.1870451, -2.5278964, 2.5348649
3: -9.8930416, -6.9045644, -9.8871117, -6.9263182, -2.6957731, 2.7153883
4: -4.5720391, -2.3973236, -4.5593243, -2.4029551, -1.5763633, 1.5798471
5: -11.0752411, -7.3699412, -11.0680733, -7.3833785, -2.5431652, 2.5421319
6: -17.6151657, -13.6024733, -17.5683270, -13.6043377, -2.8876657, 2.8360362
7: -6.4309626, -3.5945787, -6.4200583, -3.6006618, -2.1850200, 2.1736746
8: -2.0575070, 0.1833611, -2.0364141, 0.1771607, -1.7785182, 1.7609050
9: 2.4164243, 5.1634159, 2.4229517, 5.1593704, -2.2954931, 2.2860534

Time for backsubstitution: 22.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3583334
time: 4.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3583344
time: 4.51 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.2906151, -10.3198061, -14.3055668, -10.2878733, -2.5325003, 2.5077128
1: -12.4902802, -8.9498072, -12.4945412, -8.9114246, -2.5829887, 2.5529456
2: -13.3974428, -10.1855984, -13.4044533, -10.1190605, -2.5747833, 2.5342174
3: -9.8801508, -6.9058275, -9.9375839, -6.8926001, -2.7138996, 2.7549288
4: -4.5602002, -2.4147863, -4.5679855, -2.3987148, -1.5695097, 1.5850540
5: -11.0646238, -7.3681016, -11.1309242, -7.3625088, -2.5445747, 2.6047208
6: -17.5740166, -13.6286755, -17.6125069, -13.6119022, -2.8670545, 2.8855920
7: -6.4302216, -3.6008029, -6.4406657, -3.5596161, -2.2227905, 2.1854522
8: -2.0340743, 0.1711874, -2.0600452, 0.1783490, -1.7700644, 1.7794690
9: 2.4267797, 5.1590261, 2.4149947, 5.1647029, -2.2919908, 2.2903986

Time for backsubstitution: 22.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3584036
time: 5.83 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3584040
time: 6.67 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.3440590, -10.2844648, -14.3110409, -10.2687244, -2.5782943, 2.5339117
1: -12.5052376, -8.9332829, -12.4970121, -8.9031906, -2.6074772, 2.5673571
2: -13.4136257, -10.1668463, -13.4117126, -10.1156034, -2.5939360, 2.5632386
3: -9.8948431, -6.8911209, -9.9435387, -6.8909063, -2.7310343, 2.7698135
4: -4.5728326, -2.3956444, -4.5683622, -2.3898253, -1.5926571, 1.6020170
5: -11.0779285, -7.3603992, -11.1362400, -7.3613510, -2.5680780, 2.6179311
6: -17.6217308, -13.6018200, -17.6162872, -13.5969315, -2.9175434, 2.9064760
7: -6.4381695, -3.5918012, -6.4423928, -3.5563650, -2.2380986, 2.1997409
8: -2.0595121, 0.1869874, -2.0637112, 0.1857471, -1.7951226, 1.7980053
9: 2.4132409, 5.1638780, 2.4093289, 5.1654444, -2.3066778, 2.3013792

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3597394
time: 4.34 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3597403
time: 4.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.67 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3482328
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3482327
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3495646
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3495653
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3496336
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3557159, upper bound: 1.3496332
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3509716
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3579837, upper bound: 1.3509694
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3570063
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3570042
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3583334
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3583344
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3584036
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3487028, upper bound: 1.3584040
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3597394
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 9, lower bound: -1.3509700, upper bound: 1.3597403

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.2800293, -10.3245754, -14.2791634, -10.3114624, -2.4892459, 2.4747577
1: -12.4714394, -8.9589453, -12.4710808, -8.9552631, -2.5175991, 2.5131402
2: -13.3864975, -10.1920338, -13.3861876, -10.1930494, -2.4994955, 2.4972467
3: -9.8741417, -6.9207358, -9.8769474, -6.9298434, -2.6706748, 2.6806979
4: -4.5449238, -2.4245219, -4.5444694, -2.4197726, -1.5436373, 1.5390649
5: -11.0581093, -7.3824072, -11.0591984, -7.3893299, -2.5086937, 2.5179315
6: -17.5575600, -13.6354122, -17.5547237, -13.6253967, -2.8106337, 2.7985649
7: -6.4186945, -3.6120958, -6.4139938, -3.6123056, -2.1563630, 2.1540406
8: -2.0269084, 0.1624937, -2.0277481, 0.1647005, -1.7348347, 1.7349832
9: 2.4411793, 5.1450615, 2.4398379, 5.1451597, -2.2560377, 2.2563393

Time for backsubstitution: 22.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543328, upper bound: 1.3482332
time: 5.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543328, upper bound: 1.3482319
time: 9.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.2800293, -10.3245754, -14.2838402, -10.3071461, -2.4934244, 2.4788909
1: -12.4714394, -8.9589453, -12.4872990, -8.9483471, -2.5243111, 2.5292792
2: -13.3864975, -10.1920338, -13.3904705, -10.1903486, -2.5023727, 2.5023642
3: -9.8741417, -6.9207358, -9.8811760, -6.9282269, -2.6734395, 2.6851583
4: -4.5449238, -2.4245219, -4.5589838, -2.4117446, -1.5512915, 1.5534270
5: -11.0581093, -7.3824072, -11.0629244, -7.3845358, -2.5134840, 2.5220780
6: -17.5575600, -13.6354122, -17.5647926, -13.6193142, -2.8167992, 2.8092785
7: -6.4186945, -3.6120958, -6.4183016, -3.6038113, -2.1651416, 2.1586986
8: -2.0269084, 0.1624937, -2.0329933, 0.1697607, -1.7401233, 1.7404673
9: 2.4411793, 5.1450615, 2.4286103, 5.1586695, -2.2695947, 2.2675703

Time for backsubstitution: 22.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543328, upper bound: 1.3482328
time: 4.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3543328, upper bound: 1.3482328
time: 4.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -14.3333693, -10.2892113, -14.2846222, -10.2923164, -2.5549459, 2.5007982
1: -12.4863548, -8.9424267, -12.4735632, -8.9472046, -2.5419989, 2.5274615
2: -13.4027281, -10.1735001, -13.3935089, -10.1897478, -2.5184784, 2.5259924
3: -9.8888168, -6.9061794, -9.8828802, -6.9279366, -2.6877260, 2.7015257
4: -4.5575218, -2.4053645, -4.5448055, -2.4109793, -1.5666678, 1.5578413
5: -11.0715113, -7.3747234, -11.0643501, -7.3881721, -2.5320015, 2.5329237
6: -17.6050568, -13.6085625, -17.5582428, -13.6104202, -2.8749151, 2.8191643
7: -6.4266610, -3.6030998, -6.4157515, -3.6091785, -2.1715574, 2.1682973
8: -2.0522833, 0.1782985, -2.0311775, 0.1721015, -1.7677641, 1.7534513
9: 2.4277053, 5.1499014, 2.4341893, 5.1458578, -2.2707357, 2.2673087

Time for backsubstitution: 23.25 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.77 + 546.00 = 602.78 seconds
