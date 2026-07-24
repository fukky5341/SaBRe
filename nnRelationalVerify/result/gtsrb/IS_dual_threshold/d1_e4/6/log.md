## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 1800 seconds
Split limit: 100
Threshold: 5.2117594236


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5712433, 17.5712471)
1: (-39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.9048615, 11.9048615)
2: (-27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0282974, 11.0282974)
3: (-31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9663391, 10.9663391)
4: (-29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2107773, 14.2107849)
5: (-31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1767807, 12.1767807)
6: (-14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6564217, 11.6564217)
7: (-46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0449066, 12.0449066)
8: (-41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7334137, 10.7334156)
9: (-24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4951477, 16.4951553)
10: (-52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1786499, 17.1786499)
11: (-47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0818787, 15.0818825)
12: (-13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4550247, 15.4550285)
13: (-9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3225098, 16.3225098)
14: (-86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9789429, 19.9789467)
15: (-29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1688499, 12.1688499)
16: (-43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2640076, 16.2640114)
17: (-99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1865158, 22.1865158)
18: (-17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7049484, 13.7049484)
19: (-21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4383926, 12.4383965)
20: (-8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471)
21: (-30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1385345, 16.1385384)
22: (-24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1721649, 12.1721611)
23: (-16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1715317, 14.1715317)
24: (-8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7902565, 12.7902565)
25: (-4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1914444, 14.1914444)
26: (-23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3224030, 18.3224030)
27: (-17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9048843, 12.9048805)
28: (-3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9832230, 15.9832306)
29: (-41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5519714, 14.5519714)
30: (-11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7511139, 17.7511139)
31: (-22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2867203, 15.2867241)
32: (-3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2883644, 11.2883644)
33: (10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3229675, 16.3229675)
34: (11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4497757, 11.4497757)
35: (22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3710098, 11.3710098)
36: (17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4081650, 12.4081612)
37: (7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7833710, 16.7833786)
38: (6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4383469, 14.4383507)
39: (5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2389374, 16.2389450)
40: (0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6421280, 12.6421318)
41: (-4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9802513, 10.9802513)
42: (-27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5340080, 11.5340080)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.63 + 19.16 = 21.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 35, lower bound: -5.2169764, upper bound: 5.2169764

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 753

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2152245, upper bound: 5.1961312
time: 17.54 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2165370, upper bound: 5.2165368
time: 8.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 26.19 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 26.19
Output dim: 35, lower bound: -5.2152245, upper bound: 5.1961312
IS_B2, status: Status.UNKNOWN, split count: 1, time: 26.19
Output dim: 35, lower bound: -5.2165370, upper bound: 5.2165368

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -57.6421928, -32.6394768, -57.6235428, -32.6609764, -17.5111427, 17.5110245
1: -39.1879921, -20.2136459, -39.1754150, -20.2285366, -11.8657417, 11.8675041
2: -27.2299309, -11.1675386, -27.2222710, -11.1784019, -10.9985962, 11.0014305
3: -31.5753593, -14.0832863, -31.5708923, -14.0887280, -10.9458084, 10.9452095
4: -29.3894310, -8.6425858, -29.3789482, -8.6552820, -14.1769867, 14.1784363
5: -31.7466164, -13.5465975, -31.7406578, -13.5576096, -12.1468582, 12.1511536
6: -14.8823872, 2.8857493, -14.8661537, 2.8718519, -11.6140213, 11.6144524
7: -46.6406975, -25.5550842, -46.6229591, -25.5760803, -11.9917679, 11.9947357
8: -41.4304657, -19.8660717, -41.4137650, -19.8958912, -10.6595764, 10.6719818
9: -24.2787724, -5.1367278, -24.2704887, -5.1464128, -16.4650040, 16.4666061
10: -52.0960274, -29.6783791, -52.0736923, -29.7097778, -17.0980682, 17.1039276
11: -47.9109268, -27.1021080, -47.8969727, -27.1155319, -15.0537262, 15.0664406
12: -13.3544388, 5.9189639, -13.3472786, 5.9114561, -15.4228401, 15.4254112
13: -9.2609711, 9.7470665, -9.2433462, 9.7360630, -16.2747726, 16.2675171
14: -86.0896759, -59.5350418, -86.0445023, -59.5773849, -19.8689957, 19.8665352
15: -29.5672836, -11.9239445, -29.5612640, -11.9339390, -12.1383629, 12.1427536
16: -43.3744125, -22.5743771, -43.3609581, -22.5930710, -16.2073441, 16.2198410
17: -99.9721909, -70.0551834, -99.9413147, -70.0888367, -22.1015091, 22.1212463
18: -17.7555313, 3.4660516, -17.7530479, 3.4609632, -13.6867065, 13.6915092
19: -21.0151863, -6.4551973, -21.0023613, -6.4688601, -12.4036140, 12.4150887
20: -8.1900215, 5.5783615, -8.1839132, 5.5688825, -13.7589035, 13.7622747
21: -30.4788017, -12.1579494, -30.4612808, -12.1741600, -16.0981140, 16.1135597
22: -24.8010368, -8.3599758, -24.7939739, -8.3646355, -12.1493835, 12.1511917
23: -16.8752937, 0.1437353, -16.8649883, 0.1282449, -14.1314774, 14.1393890
24: -8.0106773, 6.9022188, -8.0058069, 6.8979292, -12.7702255, 12.7708206
25: -4.5866909, 11.7229424, -4.5770540, 11.7139950, -14.1636505, 14.1651268
26: -23.0520878, -1.5660565, -23.0451298, -1.5759423, -18.2823868, 18.2951889
27: -17.8048954, -3.7891860, -17.7997284, -3.7937031, -12.8834305, 12.8871918
28: -3.3255055, 16.1684990, -3.3179371, 16.1629143, -15.9594116, 15.9579163
29: -41.7338715, -23.3586655, -41.7279739, -23.3657913, -14.5309372, 14.5365028
30: -11.7961082, 7.2517490, -11.7923317, 7.2484918, -17.7298737, 17.7316513
31: -22.9005394, -4.3959470, -22.8949623, -4.4095063, -15.2448654, 15.2556496
32: -3.7865028, 10.6015882, -3.7725816, 10.5959988, -11.2596588, 11.2545509
33: 10.5397434, 30.8868542, 10.5713205, 30.8728676, -16.2603607, 16.2450943
34: 11.2949333, 28.9862785, 11.3212461, 28.9651146, -11.3875580, 11.3833427
35: 22.9544334, 40.4924393, 22.9879208, 40.4731636, -11.3020363, 11.2884750
36: 17.9838924, 34.5315170, 18.0191021, 34.5168724, -12.3435669, 12.3246040
37: 7.8717966, 28.1318207, 7.8771296, 28.1278687, -16.7596130, 16.7596970
38: 6.6185808, 26.5953064, 6.6501174, 26.5859184, -14.3768196, 14.3553810
39: 5.7309084, 25.9544830, 5.7588549, 25.9509506, -16.1877747, 16.1673965
40: 0.6023083, 19.8642559, 0.6151781, 19.8538361, -12.6021423, 12.6027412
41: -4.0797386, 9.0897808, -4.0702515, 9.0805893, -10.9545593, 10.9537773
42: -27.5850220, -10.8508787, -27.5826817, -10.8590479, -11.5160294, 11.5218925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 629

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1939993, upper bound: 5.1947732
time: 6.10 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2148209, upper bound: 5.1957279
time: 69.48 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -57.6477814, -32.6078339, -57.6475639, -32.6085587, -17.5332298, 17.5698929
1: -39.1898766, -20.1922874, -39.1897659, -20.1927719, -11.8793716, 11.9034653
2: -27.2320366, -11.1519089, -27.2318897, -11.1522522, -11.0158653, 11.0275803
3: -31.5783234, -14.0758181, -31.5775928, -14.0760489, -10.9653587, 10.9653625
4: -29.3918991, -8.6250563, -29.3916779, -8.6254644, -14.1900787, 14.2098961
5: -31.7480373, -13.5297308, -31.7479630, -13.5301342, -12.1748314, 12.1760597
6: -14.9047918, 2.8873687, -14.9042854, 2.8872652, -11.6529999, 11.6183090
7: -46.6422806, -25.5249729, -46.6421623, -25.5256004, -12.0149078, 12.0439491
8: -41.4314156, -19.8238888, -41.4313278, -19.8247528, -10.6950150, 10.7321587
9: -24.2805824, -5.1231542, -24.2804737, -5.1237421, -16.4793625, 16.4904251
10: -52.0965385, -29.6346760, -52.0964966, -29.6356621, -17.1133995, 17.1746674
11: -47.9129181, -27.0817680, -47.9127350, -27.0823059, -15.0773468, 15.0783005
12: -13.3637142, 5.9230442, -13.3633928, 5.9218154, -15.4534149, 15.4532089
13: -9.2803259, 9.7614651, -9.2797394, 9.7609177, -16.3186111, 16.3200150
14: -86.0950470, -59.4744797, -86.0947571, -59.4756775, -19.8946991, 19.9768448
15: -29.5715714, -11.9095106, -29.5713921, -11.9099083, -12.1614418, 12.1675034
16: -43.3782234, -22.5441360, -43.3780479, -22.5448952, -16.2462006, 16.2611618
17: -99.9763107, -70.0079269, -99.9757690, -70.0088501, -22.1401672, 22.1846695
18: -17.7609005, 3.4688952, -17.7607231, 3.4669476, -13.7063713, 13.7023430
19: -21.0213108, -6.4336634, -21.0210476, -6.4340143, -12.4287109, 12.4367981
20: -8.1966448, 5.5915527, -8.1963387, 5.5912247, -13.7878695, 13.7878914
21: -30.4832878, -12.1335964, -30.4830074, -12.1340866, -16.1282501, 16.1354675
22: -24.8075066, -8.3546381, -24.8072205, -8.3549404, -12.1709900, 12.1699638
23: -16.8795319, 0.1660688, -16.8793640, 0.1655033, -14.1515961, 14.1699104
24: -8.0179510, 6.9063330, -8.0177116, 6.9058280, -12.7919464, 12.7867775
25: -4.5947208, 11.7346287, -4.5944395, 11.7343388, -14.1852417, 14.1897202
26: -23.0606346, -1.5582230, -23.0603561, -1.5603030, -18.3363113, 18.3172226
27: -17.8109512, -3.7843366, -17.8108063, -3.7848933, -12.9092255, 12.9014626
28: -3.3351674, 16.1734619, -3.3347499, 16.1723633, -15.9803543, 15.9806290
29: -41.7371101, -23.3488674, -41.7369270, -23.3491383, -14.5505066, 14.5501175
30: -11.7991228, 7.2566581, -11.7982359, 7.2563810, -17.7513809, 17.7470627
31: -22.9119530, -4.3765745, -22.9116173, -4.3775568, -15.2794800, 15.2840309
32: -3.8055558, 10.6030436, -3.8050721, 10.6028814, -11.2840996, 11.2727203
33: 10.4944916, 30.8884201, 10.4955959, 30.8882942, -16.3209839, 16.2886658
34: 11.2571335, 28.9885178, 11.2578678, 28.9881287, -11.4480743, 11.4233093
35: 22.9058304, 40.4928207, 22.9068527, 40.4926987, -11.3695755, 11.3320694
36: 17.9336548, 34.5322418, 17.9346924, 34.5320511, -12.4065628, 12.3774643
37: 7.8627629, 28.1336823, 7.8630896, 28.1316872, -16.7807159, 16.7805672
38: 6.5737424, 26.5973568, 6.5746603, 26.5972042, -14.4366608, 14.4220772
39: 5.6913548, 25.9544945, 5.6923051, 25.9528236, -16.2397690, 16.2352295
40: 0.5839858, 19.8676109, 0.5843635, 19.8673630, -12.6390381, 12.6184158
41: -4.0925803, 9.0920830, -4.0922842, 9.0918179, -10.9792747, 10.9702950
42: -27.5882721, -10.8448505, -27.5879803, -10.8458557, -11.5312996, 11.5316010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 629

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2152091, upper bound: 5.1953583
time: 17.66 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2161341, upper bound: 5.2161335
time: 20.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 40.07 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 40.07
Output dim: 35, lower bound: -5.1939993, upper bound: 5.1947732
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 40.07
Output dim: 35, lower bound: -5.2148209, upper bound: 5.1957279
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 40.07
Output dim: 35, lower bound: -5.2152091, upper bound: 5.1953583
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 40.07
Output dim: 35, lower bound: -5.2161341, upper bound: 5.2161335

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -57.6415176, -32.6397285, -57.6224289, -32.6613770, -17.5086823, 17.5070076
1: -39.1872940, -20.2139645, -39.1743393, -20.2290764, -11.8674316, 11.8648758
2: -27.2297192, -11.1677856, -27.2219048, -11.1788101, -10.9979210, 10.9974899
3: -31.5738907, -14.0836229, -31.5683937, -14.0892467, -10.9388618, 10.9383240
4: -29.3886948, -8.6428757, -29.3776627, -8.6557302, -14.1757736, 14.1511765
5: -31.7463951, -13.5475893, -31.7402630, -13.5592880, -12.1420746, 12.1480217
6: -14.8763094, 2.8856235, -14.8563232, 2.8716455, -11.6092529, 11.5987396
7: -46.6403427, -25.5554924, -46.6223679, -25.5767708, -11.9886246, 11.9933014
8: -41.4302063, -19.8664417, -41.4133644, -19.8964748, -10.6584282, 10.6635017
9: -24.2778358, -5.1369934, -24.2688580, -5.1467695, -16.4647293, 16.4642563
10: -52.0947723, -29.6789303, -52.0716400, -29.7106934, -17.0960045, 17.0522461
11: -47.9107437, -27.1029987, -47.8966179, -27.1170559, -15.0478668, 15.0740509
12: -13.3531466, 5.9186654, -13.3451214, 5.9109559, -15.4211349, 15.3895874
13: -9.2603855, 9.7441368, -9.2424088, 9.7315092, -16.2883148, 16.2552414
14: -86.0873795, -59.5351410, -86.0406418, -59.5775528, -19.8450775, 19.8745804
15: -29.5664558, -11.9240952, -29.5598412, -11.9341364, -12.1372871, 12.1352310
16: -43.3736267, -22.5748577, -43.3596077, -22.5938797, -16.2048111, 16.2145004
17: -99.9706039, -70.0554047, -99.9387741, -70.0891495, -22.0727997, 22.1517029
18: -17.7541637, 3.4658754, -17.7508812, 3.4606638, -13.6669502, 13.7030182
19: -21.0149956, -6.4571605, -21.0020370, -6.4719172, -12.3983574, 12.4141922
20: -8.1882963, 5.5779533, -8.1811476, 5.5682516, -13.7565479, 13.7591009
21: -30.4785042, -12.1600714, -30.4608231, -12.1769524, -16.0804138, 16.1121445
22: -24.8003845, -8.3607502, -24.7928219, -8.3658991, -12.1440811, 12.1589279
23: -16.8750935, 0.1427283, -16.8646812, 0.1265048, -14.1089172, 14.1381035
24: -8.0105114, 6.8994017, -8.0055799, 6.8937988, -12.7612915, 12.7756500
25: -4.5864916, 11.7217884, -4.5767164, 11.7124157, -14.1533203, 14.1638832
26: -23.0506573, -1.5662589, -23.0426579, -1.5762467, -18.2659683, 18.3115234
27: -17.8011169, -3.7893357, -17.7937088, -3.7939563, -12.8741531, 12.8944054
28: -3.3251133, 16.1679382, -3.3173101, 16.1619606, -15.9574814, 15.9574585
29: -41.7333679, -23.3590469, -41.7271309, -23.3664036, -14.5230103, 14.5346527
30: -11.7959127, 7.2508993, -11.7919922, 7.2470427, -17.7261887, 17.7331543
31: -22.9003468, -4.3990793, -22.8946285, -4.4147081, -15.2351379, 15.2721558
32: -3.7840598, 10.6012096, -3.7695200, 10.5954208, -11.2576904, 11.2186623
33: 10.5401955, 30.8849201, 10.5720282, 30.8704357, -16.2521820, 16.2360840
34: 11.3003445, 28.9861298, 11.3304491, 28.9648724, -11.3830528, 11.3724098
35: 22.9549274, 40.4915237, 22.9887009, 40.4716187, -11.2548256, 11.2868347
36: 17.9847755, 34.5309448, 18.0205612, 34.5158997, -12.3378067, 12.3152542
37: 7.8724785, 28.1314583, 7.8782468, 28.1271915, -16.7254944, 16.7527771
38: 6.6239338, 26.5950699, 6.6592131, 26.5855141, -14.3700676, 14.3453064
39: 5.7314029, 25.9517860, 5.7596493, 25.9464149, -16.1905365, 16.1602478
40: 0.6041131, 19.8641472, 0.6181030, 19.8536587, -12.6007004, 12.5687447
41: -4.0775375, 9.0895538, -4.0664892, 9.0801601, -10.9615936, 10.9429169
42: -27.5828476, -10.8511906, -27.5789394, -10.8595200, -11.5151176, 11.5204659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 737

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2140843, upper bound: 5.1765459
time: 15.24 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2145072, upper bound: 5.1951140
time: 16.70 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -57.6295128, -32.6162567, -57.6379662, -32.6097183, -17.5150795, 17.5548935
1: -39.1786232, -20.1993523, -39.1836090, -20.1938934, -11.8665543, 11.8859787
2: -27.2242584, -11.1559200, -27.2278004, -11.1528254, -11.0077286, 11.0191956
3: -31.5731201, -14.0811481, -31.5751286, -14.0774193, -10.9564590, 10.9569206
4: -29.3624268, -8.6512814, -29.3748817, -8.6266727, -14.1599884, 14.1678543
5: -31.7451496, -13.5374289, -31.7473297, -13.5319996, -12.1682472, 12.1667557
6: -14.8883619, 2.8865471, -14.8948727, 2.8866210, -11.6340218, 11.6059799
7: -46.6320419, -25.5300102, -46.6367912, -25.5263100, -12.0067902, 12.0384369
8: -41.4220352, -19.8310032, -41.4260139, -19.8256950, -10.6854973, 10.7220516
9: -24.2517204, -5.1480761, -24.2637062, -5.1253209, -16.4495163, 16.4502182
10: -52.0489120, -29.6808434, -52.0684662, -29.6381264, -17.0641747, 17.1016769
11: -47.9041519, -27.0946522, -47.9105377, -27.0893669, -15.0466385, 15.0620613
12: -13.3265057, 5.8955550, -13.3428793, 5.9193540, -15.4168892, 15.4076881
13: -9.2750320, 9.7448292, -9.2783451, 9.7537689, -16.3140182, 16.2964668
14: -86.0256195, -59.5219383, -86.0556793, -59.4764938, -19.8405457, 19.9399719
15: -29.5507965, -11.9270372, -29.5597401, -11.9113560, -12.1399918, 12.1390152
16: -43.3582458, -22.5551682, -43.3671722, -22.5466347, -16.2265625, 16.2400284
17: -99.9393921, -70.0241241, -99.9559631, -70.0095825, -22.0964813, 22.1684265
18: -17.7317390, 3.4521971, -17.7450562, 3.4661942, -13.6786957, 13.6865120
19: -21.0076714, -6.4528275, -21.0195122, -6.4451571, -12.3943710, 12.4130173
20: -8.1834202, 5.5872965, -8.1913662, 5.5889874, -13.7724075, 13.7786627
21: -30.4585381, -12.1657000, -30.4800873, -12.1525993, -16.0760117, 16.0971718
22: -24.8006935, -8.3568983, -24.8048534, -8.3558617, -12.1530914, 12.1645737
23: -16.8497391, 0.1304635, -16.8776531, 0.1442187, -14.0965500, 14.1316338
24: -8.0110188, 6.8972478, -8.0154495, 6.9008093, -12.7599487, 12.7691383
25: -4.5655861, 11.7020655, -4.5927978, 11.7152653, -14.1341476, 14.1537590
26: -23.0317745, -1.5659401, -23.0468445, -1.5613637, -18.3038101, 18.3049011
27: -17.7986488, -3.7854123, -17.8052120, -3.7857752, -12.8951721, 12.9017868
28: -3.3149252, 16.1529160, -3.3330746, 16.1603851, -15.9462433, 15.9580688
29: -41.7257729, -23.3607941, -41.7352409, -23.3559418, -14.5332642, 14.5374298
30: -11.7770329, 7.2299051, -11.7960558, 7.2414274, -17.7104721, 17.7182007
31: -22.9026833, -4.3817959, -22.9093781, -4.3802013, -15.2405319, 15.2695236
32: -3.7660940, 10.5727577, -3.7816632, 10.6004820, -11.2414551, 11.2175179
33: 10.5229578, 30.8578911, 10.4987106, 30.8703270, -16.3046265, 16.2687683
34: 11.2709866, 28.9865608, 11.2646980, 28.9867573, -11.4320641, 11.4128265
35: 22.9447784, 40.4563179, 22.9091015, 40.4711876, -11.3164330, 11.2981796
36: 17.9396992, 34.5266724, 17.9359016, 34.5290222, -12.3912926, 12.3664932
37: 7.9070406, 28.0788803, 7.8661084, 28.0994053, -16.7390976, 16.7415199
38: 6.5906925, 26.5934868, 6.5817032, 26.5957088, -14.4165039, 14.4098740
39: 5.6974454, 25.9503479, 5.6946654, 25.9511719, -16.2303772, 16.2217636
40: 0.6134796, 19.8479633, 0.6005707, 19.8666725, -12.6085701, 12.5808792
41: -4.0882311, 9.0898180, -4.0905981, 9.0895920, -10.9653244, 10.9453125
42: -27.5823040, -10.8477268, -27.5857067, -10.8472519, -11.5210762, 11.5243530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 737

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1911306, upper bound: 5.1945890
time: 5.22 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2149122, upper bound: 5.1950598
time: 17.36 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -57.6466675, -32.6082382, -57.6469116, -32.6087952, -17.5291901, 17.5674400
1: -39.1887932, -20.1927891, -39.1890488, -20.1930809, -11.8767357, 11.9051666
2: -27.2316704, -11.1523066, -27.2316780, -11.1525135, -11.0119324, 11.0269012
3: -31.5758362, -14.0762920, -31.5761223, -14.0763626, -10.9584732, 10.9583931
4: -29.3906097, -8.6255312, -29.3909168, -8.6257229, -14.1627960, 14.2086716
5: -31.7476425, -13.5314007, -31.7477150, -13.5311327, -12.1716690, 12.1712723
6: -14.8949471, 2.8871818, -14.8982353, 2.8871307, -11.6372643, 11.6135483
7: -46.6417198, -25.5256386, -46.6418152, -25.5259647, -12.0134926, 12.0407944
8: -41.4309998, -19.8244534, -41.4310989, -19.8250580, -10.6865540, 10.7310143
9: -24.2789536, -5.1235032, -24.2795601, -5.1239834, -16.4770432, 16.4901428
10: -52.0944443, -29.6355419, -52.0952911, -29.6362228, -17.0617142, 17.1725769
11: -47.9125938, -27.0833530, -47.9125366, -27.0832100, -15.0849380, 15.0724411
12: -13.3615627, 5.9225321, -13.3620882, 5.9214993, -15.4176064, 15.4515076
13: -9.2794075, 9.7568932, -9.2791643, 9.7580061, -16.3063431, 16.3335304
14: -86.0912247, -59.4746399, -86.0924225, -59.4758072, -19.9027023, 19.9529419
15: -29.5701332, -11.9097118, -29.5705853, -11.9100533, -12.1539574, 12.1663895
16: -43.3768768, -22.5449905, -43.3772659, -22.5454025, -16.2408981, 16.2586517
17: -99.9737625, -70.0082092, -99.9741669, -70.0090332, -22.1706390, 22.1559448
18: -17.7587166, 3.4686112, -17.7593384, 3.4667883, -13.7178459, 13.6825676
19: -21.0209904, -6.4367199, -21.0208416, -6.4359474, -12.4278259, 12.4315262
20: -8.1938906, 5.5909152, -8.1946135, 5.5908275, -13.7847176, 13.7855282
21: -30.4828186, -12.1363783, -30.4827080, -12.1362238, -16.1268387, 16.1177750
22: -24.8063545, -8.3558683, -24.8065453, -8.3556986, -12.1787262, 12.1646690
23: -16.8792229, 0.1643460, -16.8791962, 0.1645126, -14.1502914, 14.1473236
24: -8.0177069, 6.9022579, -8.0175762, 6.9029961, -12.7967873, 12.7778091
25: -4.5943894, 11.7330503, -4.5942364, 11.7331476, -14.1839905, 14.1793900
26: -23.0582161, -1.5585241, -23.0588875, -1.5604672, -18.3526611, 18.3008347
27: -17.8049583, -3.7845817, -17.8070087, -3.7850418, -12.9164734, 12.8922272
28: -3.3345499, 16.1725082, -3.3343265, 16.1718140, -15.9799118, 15.9786682
29: -41.7363052, -23.3494720, -41.7364311, -23.3494873, -14.5486603, 14.5421753
30: -11.7987890, 7.2551799, -11.7980309, 7.2555180, -17.7528839, 17.7433624
31: -22.9116535, -4.3817940, -22.9114075, -4.3807259, -15.2959747, 15.2743149
32: -3.8024931, 10.6024704, -3.8026471, 10.6024952, -11.2482071, 11.2707634
33: 10.4952230, 30.8860130, 10.4960346, 30.8863487, -16.3119888, 16.2804718
34: 11.2663498, 28.9882679, 11.2632627, 28.9879570, -11.4371338, 11.4188004
35: 22.9066048, 40.4912720, 22.9073296, 40.4917908, -11.3679352, 11.2848663
36: 17.9351082, 34.5312538, 17.9355888, 34.5314865, -12.3972244, 12.3717117
37: 7.8638802, 28.1330376, 7.8637776, 28.1312981, -16.7738037, 16.7464828
38: 6.5828495, 26.5969810, 6.5800266, 26.5969677, -14.4266090, 14.4152908
39: 5.6921349, 25.9499435, 5.6927485, 25.9500885, -16.2326279, 16.2379837
40: 0.5868278, 19.8674011, 0.5861816, 19.8672676, -12.6050148, 12.6170006
41: -4.0888271, 9.0916681, -4.0900693, 9.0915756, -10.9683952, 10.9773254
42: -27.5845413, -10.8452930, -27.5857544, -10.8460999, -11.5298538, 11.5306931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 737

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1920705, upper bound: 5.2153948
time: 5.54 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2158390, upper bound: 5.2158386
time: 9.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.42 seconds
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 17.42
Output dim: 35, lower bound: -5.2140843, upper bound: 5.1765459
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 17.42
Output dim: 35, lower bound: -5.2145072, upper bound: 5.1951140
IS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 17.42
Output dim: 35, lower bound: -5.1911306, upper bound: 5.1945890
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 17.42
Output dim: 35, lower bound: -5.2149122, upper bound: 5.1950598
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 17.42
Output dim: 35, lower bound: -5.1920705, upper bound: 5.2153948
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 17.42
Output dim: 35, lower bound: -5.2158390, upper bound: 5.2158386

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -57.6380272, -32.6575623, -57.6164856, -32.6908607, -17.4735031, 17.4824562
1: -39.1862907, -20.2282715, -39.1725883, -20.2534008, -11.8419380, 11.8487930
2: -27.2285194, -11.1771011, -27.2198200, -11.1946173, -10.9797630, 10.9852524
3: -31.5718956, -14.0890026, -31.5651302, -14.0983467, -10.9250488, 10.9270134
4: -29.3875790, -8.6539736, -29.3757019, -8.6745968, -14.1554108, 14.1377487
5: -31.7455349, -13.5579519, -31.7387543, -13.5768614, -12.1236649, 12.1363106
6: -14.8628016, 2.8847971, -14.8332596, 2.8701611, -11.5934181, 11.5733223
7: -46.6394424, -25.5735931, -46.6208420, -25.6075211, -11.9569511, 11.9737549
8: -41.4295769, -19.8898945, -41.4123001, -19.9362144, -10.6177711, 10.6387959
9: -24.2767296, -5.1449709, -24.2670784, -5.1603165, -16.4470367, 16.4507141
10: -52.0944824, -29.7022400, -52.0710297, -29.7503185, -17.0526924, 17.0258560
11: -47.9095840, -27.1125774, -47.8946495, -27.1329041, -15.0366974, 15.0629387
12: -13.3479271, 5.9144077, -13.3363361, 5.9038057, -15.3992081, 15.3672409
13: -9.2497597, 9.7336521, -9.2243490, 9.7136345, -16.2590790, 16.2257271
14: -86.0839615, -59.5717239, -86.0348358, -59.6398544, -19.7796936, 19.8330727
15: -29.5639610, -11.9333639, -29.5556240, -11.9498930, -12.1179886, 12.1205978
16: -43.3713150, -22.5921574, -43.3556061, -22.6227150, -16.1719437, 16.1904259
17: -99.9682617, -70.0880814, -99.9349060, -70.1448212, -22.0187988, 22.1157074
18: -17.7509155, 3.4629798, -17.7454109, 3.4557943, -13.6563721, 13.6900291
19: -21.0112247, -6.4663777, -20.9955444, -6.4870300, -12.3833427, 12.3999290
20: -8.1839952, 5.5725622, -8.1736431, 5.5595264, -13.7435217, 13.7462053
21: -30.4757919, -12.1712866, -30.4561214, -12.1948595, -16.0637054, 16.0956879
22: -24.7966557, -8.3657398, -24.7864265, -8.3743286, -12.1300163, 12.1428680
23: -16.8724556, 0.1315552, -16.8600311, 0.1080025, -14.0898209, 14.1231079
24: -8.0058489, 6.8972435, -7.9976521, 6.8903513, -12.7495956, 12.7599983
25: -4.5817099, 11.7160549, -4.5684843, 11.7031212, -14.1400909, 14.1501007
26: -23.0451584, -1.5716186, -23.0334091, -1.5853274, -18.2449265, 18.2848663
27: -17.7976723, -3.7926159, -17.7879066, -3.7994375, -12.8642502, 12.8825836
28: -3.3191831, 16.1645546, -3.3072441, 16.1562958, -15.9444046, 15.9427948
29: -41.7312622, -23.3657780, -41.7235794, -23.3774948, -14.5114136, 14.5230255
30: -11.7929430, 7.2481694, -11.7869740, 7.2425137, -17.7144775, 17.7192612
31: -22.8930931, -4.4077501, -22.8818932, -4.4291840, -15.2157822, 15.2516747
32: -3.7721767, 10.6003647, -3.7494247, 10.5939617, -11.2437782, 11.1979446
33: 10.5670090, 30.8840256, 10.6175022, 30.8688698, -16.2244797, 16.1907272
34: 11.3227348, 28.9851532, 11.3685513, 28.9631653, -11.3591194, 11.3337250
35: 22.9819279, 40.4913712, 23.0345631, 40.4713402, -11.2278328, 11.2413826
36: 18.0124054, 34.5306473, 18.0674095, 34.5153770, -12.3108597, 12.2702675
37: 7.8792648, 28.1300697, 7.8896685, 28.1249275, -16.7108383, 16.7358704
38: 6.6507645, 26.5939980, 6.7047548, 26.5836601, -14.3378944, 14.2959328
39: 5.7550321, 25.9507294, 5.7996283, 25.9445896, -16.1628265, 16.1168823
40: 0.6188726, 19.8623371, 0.6432128, 19.8505898, -12.5800743, 12.5391541
41: -4.0696006, 9.0883045, -4.0530787, 9.0781155, -10.9510040, 10.9273605
42: -27.5788994, -10.8547316, -27.5721912, -10.8654842, -11.5058594, 11.5105209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 738

## Relational analysis of IS_B1_B2_B1_A1

### Relational analysis result of IS_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1883416, upper bound: 5.1727620
time: 24.71 seconds

## Relational analysis of IS_B1_B2_B1_A2

### Relational analysis result of IS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2137319, upper bound: 5.1757893
time: 6.32 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -57.6411057, -32.6412201, -57.6430359, -32.6635323, -17.4979248, 17.5288353
1: -39.1870880, -20.2151451, -39.1895027, -20.2304897, -11.8591232, 11.8799553
2: -27.2295361, -11.1689587, -27.2311630, -11.1801167, -10.9930344, 11.0057144
3: -31.5720444, -14.0844288, -31.5679893, -14.0896959, -10.9368515, 10.9379539
4: -29.3885021, -8.6438208, -29.3882065, -8.6562777, -14.1691895, 14.1622620
5: -31.7462921, -13.5485096, -31.7472153, -13.5606070, -12.1407166, 12.1550827
6: -14.8749523, 2.8855019, -14.8549213, 2.8880243, -11.6264076, 11.5852852
7: -46.6401711, -25.5568085, -46.6444168, -25.5784340, -11.9788361, 12.0140915
8: -41.4299774, -19.8680668, -41.4324455, -19.8980293, -10.6481743, 10.6810818
9: -24.2776184, -5.1385317, -24.2791672, -5.1479111, -16.4558868, 16.4709930
10: -52.0946350, -29.6806831, -52.0994339, -29.7106895, -17.0743866, 17.0788269
11: -47.9104767, -27.1060562, -47.9077148, -27.1212177, -15.0486603, 15.0589981
12: -13.3524227, 5.9154696, -13.3450775, 5.9113150, -15.4193268, 15.3844490
13: -9.2594261, 9.7424831, -9.2492466, 9.7299242, -16.2845078, 16.2607880
14: -86.0865173, -59.5375328, -86.0991669, -59.5811768, -19.8161392, 19.9327927
15: -29.5662441, -11.9252195, -29.5639362, -11.9351568, -12.1341248, 12.1382751
16: -43.3733635, -22.5774040, -43.3808365, -22.5975075, -16.1935196, 16.2268867
17: -99.9694214, -70.0581665, -99.9812927, -70.0923462, -22.0507202, 22.1680756
18: -17.7536507, 3.4611201, -17.7503281, 3.4557033, -13.6665611, 13.6927338
19: -21.0144920, -6.4606800, -21.0094585, -6.4766502, -12.3913155, 12.4056549
20: -8.1876612, 5.5762863, -8.1824684, 5.5668211, -13.7544823, 13.7587547
21: -30.4780006, -12.1639633, -30.4740505, -12.1820526, -16.0745773, 16.0996323
22: -24.7999344, -8.3629656, -24.7979050, -8.3679180, -12.1407814, 12.1506042
23: -16.8747330, 0.1400484, -16.8728981, 0.1239713, -14.1025467, 14.1355743
24: -8.0099106, 6.8982987, -8.0054922, 6.8938313, -12.7630768, 12.7677002
25: -4.5859394, 11.7192078, -4.5805011, 11.7099237, -14.1496429, 14.1599503
26: -23.0499191, -1.5701156, -23.0435123, -1.5776432, -18.2734680, 18.2943497
27: -17.8007507, -3.7914400, -17.7940044, -3.7955430, -12.8758545, 12.8881721
28: -3.3243508, 16.1647491, -3.3185356, 16.1589088, -15.9541779, 15.9559937
29: -41.7330017, -23.3609180, -41.7315750, -23.3677139, -14.5205383, 14.5268555
30: -11.7946568, 7.2501459, -11.7922068, 7.2473993, -17.7280884, 17.7288971
31: -22.8995476, -4.4041462, -22.8942928, -4.4218473, -15.2273712, 15.2624054
32: -3.7827005, 10.6008863, -3.7680087, 10.6012239, -11.2596626, 11.2134247
33: 10.5422297, 30.8847008, 10.5737257, 30.8902397, -16.2711639, 16.2247925
34: 11.3018904, 28.9857674, 11.3320141, 28.9897556, -11.4066048, 11.3621597
35: 22.9568691, 40.4913445, 22.9904633, 40.4958572, -11.2772751, 11.2742615
36: 17.9867458, 34.5307388, 18.0222778, 34.5322266, -12.3521233, 12.3055344
37: 7.8734426, 28.1280937, 7.8780084, 28.1238899, -16.7221832, 16.7481689
38: 6.6258669, 26.5948620, 6.6604695, 26.5973053, -14.3757935, 14.3372536
39: 5.7332749, 25.9501953, 5.7602344, 25.9470749, -16.1913757, 16.1579590
40: 0.6053615, 19.8638954, 0.6193781, 19.8692055, -12.6155167, 12.5557175
41: -4.0768795, 9.0892467, -4.0660038, 9.0888710, -10.9694672, 10.9391403
42: -27.5820122, -10.8518858, -27.5777321, -10.8557148, -11.5186195, 11.5168800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 738

## Relational analysis of IS_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1891190, upper bound: 5.1921073
time: 11.07 seconds

## Relational analysis of IS_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2142044, upper bound: 5.1948078
time: 29.31 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -57.6515274, -32.6179504, -57.6375160, -32.6111679, -17.5372314, 17.5444069
1: -39.1941299, -20.2005482, -39.1834335, -20.1950150, -11.8819466, 11.8779182
2: -27.2337532, -11.1568050, -27.2275887, -11.1539688, -11.0163193, 11.0144310
3: -31.5730629, -14.0813646, -31.5733871, -14.0781336, -10.9562798, 10.9551544
4: -29.3740005, -8.6515675, -29.3746662, -8.6275911, -14.1714478, 14.1613350
5: -31.7521172, -13.5385742, -31.7472286, -13.5328474, -12.1754532, 12.1655350
6: -14.8871698, 2.9031219, -14.8935003, 2.8865252, -11.6206856, 11.6231766
7: -46.6542435, -25.5314445, -46.6366005, -25.5276031, -12.0277863, 12.0289040
8: -41.4413071, -19.8324280, -41.4258041, -19.8272934, -10.7033195, 10.7120132
9: -24.2623119, -5.1484070, -24.2635612, -5.1264830, -16.4553757, 16.4413223
10: -52.0769081, -29.6806393, -52.0683250, -29.6398430, -17.0909462, 17.0803146
11: -47.9155998, -27.0987282, -47.9102859, -27.0923996, -15.0316162, 15.0613518
12: -13.3266373, 5.8968711, -13.3421659, 5.9166260, -15.4119797, 15.4059753
13: -9.2822628, 9.7439737, -9.2774582, 9.7525053, -16.3197708, 16.2930450
14: -86.0845184, -59.5253830, -86.0548553, -59.4788284, -19.8992462, 19.9112511
15: -29.5550079, -11.9274073, -29.5595379, -11.9122257, -12.1431427, 12.1360283
16: -43.3822517, -22.5580463, -43.3668823, -22.5491333, -16.2374344, 16.2284851
17: -99.9823227, -70.0270844, -99.9548492, -70.0122299, -22.1153870, 22.1464691
18: -17.7314491, 3.4474454, -17.7446022, 3.4614341, -13.6686478, 13.6863251
19: -21.0160217, -6.4572515, -21.0190659, -6.4486828, -12.3862991, 12.4056931
20: -8.1852560, 5.5859628, -8.1907978, 5.5873184, -13.7725744, 13.7767601
21: -30.4721317, -12.1707249, -30.4796371, -12.1565027, -16.0637741, 16.0906296
22: -24.8061638, -8.3580170, -24.8044319, -8.3574352, -12.1450806, 12.1611977
23: -16.8602371, 0.1279184, -16.8773346, 0.1416699, -14.0942841, 14.1250916
24: -8.0110722, 6.8973136, -8.0148516, 6.8996758, -12.7521439, 12.7710114
25: -4.5695596, 11.7004032, -4.5922713, 11.7133093, -14.1309204, 14.1500320
26: -23.0327702, -1.5672197, -23.0461807, -1.5652685, -18.2868958, 18.3126907
27: -17.7990456, -3.7868810, -17.8049049, -3.7878923, -12.8890381, 12.9037323
28: -3.3163228, 16.1501980, -3.3323205, 16.1572056, -15.9452972, 15.9553528
29: -41.7309837, -23.3620491, -41.7349243, -23.3578243, -14.5258331, 14.5349312
30: -11.7774601, 7.2303286, -11.7949409, 7.2406492, -17.7062836, 17.7203751
31: -22.9043083, -4.3886652, -22.9086342, -4.3850822, -15.2313004, 15.2618256
32: -3.7647185, 10.5787964, -3.7803798, 10.6001873, -11.2364044, 11.2190933
33: 10.5243912, 30.8777809, 10.5006971, 30.8701382, -16.2935791, 16.2880325
34: 11.2723799, 29.0115051, 11.2661972, 28.9863853, -11.4219856, 11.4364624
35: 22.9463215, 40.4805832, 22.9109592, 40.4709854, -11.3040695, 11.3206978
36: 17.9411755, 34.5429726, 17.9378071, 34.5288200, -12.3818359, 12.3809242
37: 7.9064455, 28.0756588, 7.8670325, 28.0960236, -16.7348404, 16.7384529
38: 6.5916023, 26.6054115, 6.5835776, 26.5954914, -14.4087677, 14.4156303
39: 5.6977334, 25.9510956, 5.6964288, 25.9496002, -16.2283020, 16.2227859
40: 0.6144662, 19.8635597, 0.6017570, 19.8664455, -12.5957947, 12.5959244
41: -4.0878701, 9.0986414, -4.0899935, 9.0893068, -10.9616051, 10.9532967
42: -27.5813656, -10.8437147, -27.5850391, -10.8480234, -11.5175056, 11.5279884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 738

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1902916, upper bound: 5.1929302
time: 13.20 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2146158, upper bound: 5.1947632
time: 5.33 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -57.6406860, -32.6390152, -57.6433792, -32.6268768, -17.5045013, 17.5319901
1: -39.1870995, -20.2174072, -39.1880646, -20.2075539, -11.8604851, 11.8794250
2: -27.2296181, -11.1687222, -27.2304668, -11.1621475, -10.9996300, 11.0086098
3: -31.5722504, -14.0856104, -31.5739841, -14.0818520, -10.9467163, 10.9441910
4: -29.3886719, -8.6446123, -29.3897915, -8.6369371, -14.1493263, 14.1881218
5: -31.7461662, -13.5491781, -31.7468643, -13.5415497, -12.1599007, 12.1525841
6: -14.8716679, 2.8857517, -14.8845844, 2.8863230, -11.6116791, 11.5974998
7: -46.6401978, -25.5567703, -46.6409302, -25.5442410, -11.9938240, 12.0088005
8: -41.4299622, -19.8646984, -41.4304504, -19.8487034, -10.6616745, 10.6898613
9: -24.2771759, -5.1371341, -24.2785149, -5.1322327, -16.4630966, 16.4718094
10: -52.0939217, -29.6753693, -52.0949326, -29.6595879, -17.0351257, 17.1291122
11: -47.9105606, -27.0996513, -47.9113541, -27.0927734, -15.0741806, 15.0594521
12: -13.3525448, 5.9146838, -13.3567734, 5.9166584, -15.3947182, 15.4288483
13: -9.2612114, 9.7388611, -9.2684984, 9.7472878, -16.2762451, 16.3037186
14: -86.0853577, -59.5372124, -86.0890350, -59.5124855, -19.8610535, 19.8872528
15: -29.5658951, -11.9260101, -29.5680847, -11.9196510, -12.1390762, 12.1466370
16: -43.3729744, -22.5756931, -43.3749847, -22.5635166, -16.2162247, 16.2246361
17: -99.9697037, -70.0642395, -99.9717560, -70.0418472, -22.1342773, 22.1009674
18: -17.7531090, 3.4637196, -17.7560730, 3.4639444, -13.7047348, 13.6718674
19: -21.0145149, -6.4529018, -21.0170784, -6.4457369, -12.4133911, 12.4148750
20: -8.1864967, 5.5815802, -8.1903028, 5.5852661, -13.7717628, 13.7718830
21: -30.4781799, -12.1551170, -30.4799957, -12.1475182, -16.1100006, 16.0981598
22: -24.7999344, -8.3644638, -24.8027744, -8.3610363, -12.1623001, 12.1501770
23: -16.8746853, 0.1452520, -16.8764954, 0.1532875, -14.1352158, 14.1277161
24: -8.0097256, 6.8986630, -8.0128841, 6.9007845, -12.7810364, 12.7659721
25: -4.5861979, 11.7231617, -4.5894423, 11.7268801, -14.1701660, 14.1659660
26: -23.0487709, -1.5677221, -23.0533180, -1.5658739, -18.3257141, 18.2792969
27: -17.7990570, -3.7901292, -17.8035679, -3.7882752, -12.9045410, 12.8822670
28: -3.3242621, 16.1667385, -3.3282917, 16.1683769, -15.9648132, 15.9651642
29: -41.7326622, -23.3609619, -41.7342987, -23.3562489, -14.5367813, 14.5297546
30: -11.7936726, 7.2505445, -11.7950106, 7.2527933, -17.7387543, 17.7315979
31: -22.8991432, -4.3965306, -22.9040794, -4.3896050, -15.2753830, 15.2537842
32: -3.7820268, 10.6009674, -3.7906003, 10.6016350, -11.2271690, 11.2562294
33: 10.5411959, 30.8844643, 10.5230484, 30.8854561, -16.2661591, 16.2525787
34: 11.3046980, 28.9865494, 11.2857819, 28.9869347, -11.3981934, 11.3947525
35: 22.9528809, 40.4909973, 22.9345036, 40.4916420, -11.3220901, 11.2577477
36: 17.9824619, 34.5307274, 17.9633751, 34.5311661, -12.3517227, 12.3445892
37: 7.8754778, 28.1307106, 7.8706341, 28.1298885, -16.7567596, 16.7317963
38: 6.6287937, 26.5950947, 6.6070127, 26.5958614, -14.3767548, 14.3829575
39: 5.7327256, 25.9481792, 5.7166247, 25.9490623, -16.1885300, 16.2099304
40: 0.6122112, 19.8643570, 0.6011009, 19.8654556, -12.5751648, 12.5961685
41: -4.0753021, 9.0896149, -4.0820661, 9.0903673, -10.9527245, 10.9666748
42: -27.5776863, -10.8512707, -27.5817661, -10.8496094, -11.5198441, 11.5212135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 738

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1748986, upper bound: 5.2130677
time: 6.56 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1912481, upper bound: 5.2150659
time: 5.13 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -57.6687431, -32.6099243, -57.6465225, -32.6102524, -17.5513573, 17.5569649
1: -39.2043266, -20.1940041, -39.1888771, -20.1942329, -11.8921394, 11.8971138
2: -27.2411613, -11.1531858, -27.2315292, -11.1535969, -11.0205078, 11.0221558
3: -31.5757504, -14.0765343, -31.5744038, -14.0771084, -10.9582634, 10.9566154
4: -29.4021587, -8.6258478, -29.3907566, -8.6266279, -14.1742592, 14.2021790
5: -31.7546139, -13.5325394, -31.7476196, -13.5319901, -12.1788597, 12.1700630
6: -14.8937616, 2.9037237, -14.8969221, 2.8870058, -11.6239510, 11.6307259
7: -46.6639137, -25.5270767, -46.6416626, -25.5272713, -12.0344925, 12.0312538
8: -41.4502411, -19.8258324, -41.4308395, -19.8266335, -10.7043495, 10.7209721
9: -24.2895470, -5.1238174, -24.2793922, -5.1251698, -16.4829102, 16.4812317
10: -52.1224747, -29.6353149, -52.0951500, -29.6379337, -17.0884552, 17.1512146
11: -47.9239960, -27.0874386, -47.9123039, -27.0862503, -15.0699158, 15.0717545
12: -13.3617258, 5.9239106, -13.3613958, 5.9187708, -15.4127197, 15.4497719
13: -9.2866325, 9.7560310, -9.2782669, 9.7567511, -16.3120956, 16.3300972
14: -86.1501236, -59.4780693, -86.0916595, -59.4781494, -19.9613876, 19.9242287
15: -29.5743599, -11.9100962, -29.5703621, -11.9109421, -12.1571121, 12.1634140
16: -43.4008789, -22.5478573, -43.3770332, -22.5479317, -16.2517548, 16.2471008
17: -100.0167236, -70.0111389, -99.9730072, -70.0117111, -22.1895599, 22.1340103
18: -17.7583656, 3.4638283, -17.7588978, 3.4620571, -13.7077866, 13.6823883
19: -21.0293159, -6.4411392, -21.0203762, -6.4394884, -12.4197273, 12.4242287
20: -8.1956978, 5.5895848, -8.1940527, 5.5891714, -13.7848692, 13.7836380
21: -30.4964371, -12.1414099, -30.4822311, -12.1401052, -16.1146088, 16.1112061
22: -24.8118057, -8.3569984, -24.8060894, -8.3572550, -12.1707115, 12.1612892
23: -16.8897038, 0.1618011, -16.8788605, 0.1619247, -14.1480179, 14.1407890
24: -8.0177822, 6.9023304, -8.0169849, 6.9018869, -12.7889404, 12.7796631
25: -4.5983620, 11.7313967, -4.5937138, 11.7311764, -14.1807632, 14.1756973
26: -23.0592270, -1.5597763, -23.0581875, -1.5643992, -18.3357010, 18.3086319
27: -17.8053341, -3.7860355, -17.8066940, -3.7871065, -12.9103394, 12.8941536
28: -3.3359513, 16.1697884, -3.3336036, 16.1686153, -15.9789352, 15.9759521
29: -41.7414703, -23.3507347, -41.7360916, -23.3513565, -14.5412064, 14.5396614
30: -11.7992153, 7.2555914, -11.7968941, 7.2547302, -17.7487183, 17.7455444
31: -22.9132442, -4.3886385, -22.9106522, -4.3856301, -15.2867126, 15.2666206
32: -3.8011036, 10.6084738, -3.8013315, 10.6022062, -11.2431564, 11.2723274
33: 10.4966955, 30.9059048, 10.4980679, 30.8861465, -16.3008957, 16.2996902
34: 11.2677526, 29.0131931, 11.2647305, 28.9876003, -11.4270477, 11.4424515
35: 22.9081726, 40.5155296, 22.9092503, 40.4915924, -11.3555603, 11.3073845
36: 17.9366093, 34.5475807, 17.9374905, 34.5312347, -12.3877220, 12.3861198
37: 7.8633237, 28.1298103, 7.8646674, 28.1279144, -16.7695618, 16.7434349
38: 6.5837488, 26.6089172, 6.5818777, 26.5967751, -14.4188614, 14.4210777
39: 5.6924844, 25.9507103, 5.6945815, 25.9485760, -16.2305374, 16.2389984
40: 0.5878820, 19.8830414, 0.5873308, 19.8670540, -12.5922279, 12.6320686
41: -4.0884395, 9.1004658, -4.0894318, 9.0912838, -10.9646950, 10.9852982
42: -27.5836048, -10.8412971, -27.5850983, -10.8469172, -11.5263023, 11.5343323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 738

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1912424, upper bound: 5.2137546
time: 16.86 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2155433, upper bound: 5.2155432
time: 63.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 82.82 seconds
IS_B1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.1883416, upper bound: 5.1727620
IS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.2137319, upper bound: 5.1757893
IS_B1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.1891190, upper bound: 5.1921073
IS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.2142044, upper bound: 5.1948078
IS_B2_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.1902916, upper bound: 5.1929302
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.2146158, upper bound: 5.1947632
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.1748986, upper bound: 5.2130677
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.1912481, upper bound: 5.2150659
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.1912424, upper bound: 5.2137546
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 82.82
Output dim: 35, lower bound: -5.2155433, upper bound: 5.2155432

## BFS IS instance: IS_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -57.6375351, -32.6590462, -57.6161728, -32.6917496, -17.4720306, 17.4378166
1: -39.1860733, -20.2295303, -39.1724701, -20.2541580, -11.8409805, 11.8222961
2: -27.2283707, -11.1780109, -27.2196903, -11.1951389, -10.9790192, 10.9699936
3: -31.5707626, -14.0896730, -31.5644493, -14.0987511, -10.9238014, 10.9252167
4: -29.3873596, -8.6551628, -29.3755760, -8.6752872, -14.1567879, 14.1151505
5: -31.7453804, -13.5588226, -31.7386761, -13.5773640, -12.1240654, 12.1326637
6: -14.8616467, 2.8845954, -14.8325310, 2.8699698, -11.5502396, 11.5671768
7: -46.6393661, -25.5748081, -46.6207848, -25.6082878, -11.9560509, 11.9495773
8: -41.4294853, -19.8914452, -41.4122086, -19.9371548, -10.6161003, 10.6103210
9: -24.2766056, -5.1459017, -24.2670193, -5.1608610, -16.4462967, 16.4308090
10: -52.0944405, -29.7040901, -52.0710068, -29.7514305, -17.0514297, 16.9822464
11: -47.9093895, -27.1141930, -47.8945351, -27.1339474, -15.0303192, 15.0824280
12: -13.3473415, 5.9083700, -13.3359613, 5.9002008, -15.3920059, 15.3639183
13: -9.2487946, 9.7324362, -9.2237406, 9.7129173, -16.2587814, 16.2191696
14: -86.0831451, -59.5739822, -86.0342941, -59.6411934, -19.7859268, 19.7702026
15: -29.5637589, -11.9344368, -29.5555096, -11.9505215, -12.1171036, 12.1002541
16: -43.3710022, -22.5940399, -43.3554153, -22.6238403, -16.1703873, 16.1758041
17: -99.9674149, -70.0900269, -99.9343414, -70.1460648, -22.0087585, 22.1298447
18: -17.7504654, 3.4599733, -17.7451706, 3.4539926, -13.6490898, 13.6942596
19: -21.0108376, -6.4698086, -20.9953079, -6.4891925, -12.3785400, 12.4110069
20: -8.1832752, 5.5722966, -8.1731968, 5.5593634, -13.7426386, 13.7454929
21: -30.4754677, -12.1738091, -30.4559040, -12.1964588, -16.0523300, 16.1082344
22: -24.7962799, -8.3689022, -24.7862129, -8.3762035, -12.1189651, 12.1546135
23: -16.8721237, 0.1299479, -16.8598328, 0.1069452, -14.0840378, 14.1360321
24: -8.0050955, 6.8970146, -7.9971914, 6.8901892, -12.7453537, 12.7677536
25: -4.5809541, 11.7159977, -4.5680504, 11.7030611, -14.1341476, 14.1548462
26: -23.0444317, -1.5719626, -23.0329819, -1.5855336, -18.2272186, 18.2976303
27: -17.7970829, -3.7929111, -17.7875538, -3.7996383, -12.8565521, 12.8870926
28: -3.3182418, 16.1643219, -3.3066704, 16.1561394, -15.9425278, 15.9426041
29: -41.7309685, -23.3680172, -41.7233810, -23.3789177, -14.5068283, 14.5358963
30: -11.7919788, 7.2477198, -11.7863111, 7.2422457, -17.7093964, 17.7207108
31: -22.8923397, -4.4122143, -22.8814812, -4.4318676, -15.2086105, 15.2546234
32: -3.7712812, 10.5996466, -3.7488651, 10.5935125, -11.2305641, 11.1936378
33: 10.5690460, 30.8838329, 10.6187134, 30.8687782, -16.2002411, 16.1940918
34: 11.3243532, 28.9847450, 11.3695068, 28.9628811, -11.3391876, 11.3334122
35: 22.9839516, 40.4911957, 23.0357933, 40.4712029, -11.2004013, 11.2402649
36: 18.0142555, 34.5305710, 18.0685043, 34.5153275, -12.2847672, 12.2690277
37: 7.8804650, 28.1296463, 7.8903646, 28.1246109, -16.7034302, 16.7356300
38: 6.6525092, 26.5937767, 6.7058067, 26.5835152, -14.3303719, 14.2928734
39: 5.7566910, 25.9497375, 5.8005853, 25.9439545, -16.1608047, 16.1155090
40: 0.6200399, 19.8619766, 0.6439066, 19.8503361, -12.5694199, 12.5385017
41: -4.0688505, 9.0879793, -4.0526371, 9.0778570, -10.9368362, 10.9332542
42: -27.5783749, -10.8551521, -27.5718842, -10.8657770, -11.4946671, 11.5179214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 754

## Relational analysis of IS_B1_B2_B1_A2_A1

### Relational analysis result of IS_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1938873, upper bound: 5.1740077
time: 6.49 seconds

## Relational analysis of IS_B1_B2_B1_A2_A2

### Relational analysis result of IS_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2132264, upper bound: 5.1746619
time: 5.72 seconds

## BFS IS instance: IS_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -57.6405563, -32.6426849, -57.6427116, -32.6644287, -17.4964600, 17.4842758
1: -39.1869011, -20.2164211, -39.1893883, -20.2312202, -11.8581734, 11.8534508
2: -27.2293739, -11.1698370, -27.2310448, -11.1806355, -10.9922714, 10.9904900
3: -31.5709324, -14.0850754, -31.5673084, -14.0900879, -10.9357033, 10.9363441
4: -29.3882580, -8.6449966, -29.3880196, -8.6569920, -14.1705704, 14.1396637
5: -31.7461586, -13.5493870, -31.7471199, -13.5611267, -12.1410980, 12.1514473
6: -14.8737946, 2.8852806, -14.8542385, 2.8878908, -11.5832024, 11.5791016
7: -46.6400757, -25.5580330, -46.6443672, -25.5791950, -11.9779549, 11.9898987
8: -41.4298477, -19.8696136, -41.4323654, -19.8989754, -10.6465225, 10.6526127
9: -24.2775345, -5.1394510, -24.2790623, -5.1484690, -16.4551849, 16.4511490
10: -52.0946426, -29.6825256, -52.0994263, -29.7118073, -17.0731735, 17.0353088
11: -47.9103165, -27.1076164, -47.9076004, -27.1221790, -15.0423203, 15.0774384
12: -13.3518257, 5.9096165, -13.3447018, 5.9078393, -15.4122162, 15.3811607
13: -9.2584991, 9.7413273, -9.2486238, 9.7292204, -16.2842407, 16.2542496
14: -86.0857773, -59.5398102, -86.0986633, -59.5824814, -19.8223419, 19.8699379
15: -29.5660267, -11.9263096, -29.5638428, -11.9357977, -12.1332626, 12.1179771
16: -43.3730431, -22.5792637, -43.3806686, -22.5986156, -16.1921387, 16.2091217
17: -99.9685974, -70.0601273, -99.9808121, -70.0934448, -22.0416794, 22.1720810
18: -17.7533035, 3.4580894, -17.7500763, 3.4538918, -13.6593475, 13.6969681
19: -21.0141220, -6.4640312, -21.0092316, -6.4786577, -12.3868370, 12.4157944
20: -8.1869659, 5.5760269, -8.1820269, 5.5666432, -13.7536087, 13.7580538
21: -30.4776802, -12.1663933, -30.4738293, -12.1835012, -16.0634384, 16.1114883
22: -24.7995453, -8.3660107, -24.7976837, -8.3697500, -12.1300659, 12.1618996
23: -16.8743858, 0.1384682, -16.8726730, 0.1230035, -14.0968018, 14.1459732
24: -8.0090961, 6.8980789, -8.0050125, 6.8936720, -12.7588654, 12.7754326
25: -4.5851717, 11.7191496, -4.5798521, 11.7098866, -14.1437531, 14.1644325
26: -23.0492058, -1.5704689, -23.0431004, -1.5778453, -18.2559128, 18.3071213
27: -17.8001690, -3.7917180, -17.7936535, -3.7957444, -12.8682175, 12.8927116
28: -3.3234057, 16.1645050, -3.3179739, 16.1587543, -15.9523315, 15.9558258
29: -41.7327309, -23.3631172, -41.7313957, -23.3690834, -14.5160904, 14.5392761
30: -11.7937193, 7.2497463, -11.7915993, 7.2471304, -17.7228317, 17.7303543
31: -22.8988419, -4.4083991, -22.8938637, -4.4244175, -15.2203751, 15.2652168
32: -3.7818298, 10.6001596, -3.7674642, 10.6007948, -11.2461319, 11.2091064
33: 10.5442762, 30.8845081, 10.5749016, 30.8901043, -16.2469864, 16.2281265
34: 11.3034344, 28.9853745, 11.3329563, 28.9894829, -11.3866615, 11.3618279
35: 22.9588451, 40.4911652, 22.9916763, 40.4957275, -11.2498589, 11.2731361
36: 17.9886093, 34.5306625, 18.0234108, 34.5321541, -12.3260269, 12.3043060
37: 7.8745451, 28.1276932, 7.8786917, 28.1236038, -16.7147827, 16.7476540
38: 6.6276016, 26.5946579, 6.6615262, 26.5971985, -14.3670616, 14.3341942
39: 5.7348871, 25.9492378, 5.7612453, 25.9465065, -16.1893845, 16.1565475
40: 0.6065435, 19.8635063, 0.6200619, 19.8689899, -12.6029434, 12.5550270
41: -4.0761557, 9.0889101, -4.0655718, 9.0886545, -10.9552879, 10.9449234
42: -27.5815010, -10.8522778, -27.5774422, -10.8559742, -11.5074272, 11.5241699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 754

## Relational analysis of IS_B1_B2_B2_A2_A1

### Relational analysis result of IS_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1945098, upper bound: 5.1934169
time: 9.66 seconds

## Relational analysis of IS_B1_B2_B2_A2_A2

### Relational analysis result of IS_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2137963, upper bound: 5.1939131
time: 35.26 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -57.6510811, -32.6194077, -57.6372604, -32.6120453, -17.5144234, 17.5000191
1: -39.1939430, -20.2017899, -39.1833305, -20.1957722, -11.8748055, 11.8514595
2: -27.2335987, -11.1576881, -27.2275257, -11.1544914, -11.0130806, 10.9992790
3: -31.5719376, -14.0820208, -31.5727043, -14.0785294, -10.9551201, 10.9537201
4: -29.3738174, -8.6527424, -29.3745461, -8.6282902, -14.1728210, 14.1407280
5: -31.7520332, -13.5394497, -31.7471771, -13.5333738, -12.1758347, 12.1628036
6: -14.8860531, 2.9029417, -14.8928461, 2.8863635, -11.5775795, 11.6099243
7: -46.6541824, -25.5326920, -46.6365623, -25.5283508, -12.0242691, 12.0047607
8: -41.4411774, -19.8339977, -41.4257278, -19.8282337, -10.6998291, 10.6835861
9: -24.2622070, -5.1493101, -24.2634811, -5.1270347, -16.4497910, 16.4211807
10: -52.0768623, -29.6824722, -52.0683174, -29.6409073, -17.0825844, 17.0368881
11: -47.9154816, -27.1002598, -47.9101791, -27.0933151, -15.0346146, 15.0796623
12: -13.3260794, 5.8911519, -13.3418016, 5.9131374, -15.4054527, 15.3969345
13: -9.2814312, 9.7427959, -9.2768984, 9.7518349, -16.3195343, 16.2877197
14: -86.0838013, -59.5276489, -86.0543976, -59.4801254, -19.9052734, 19.8555145
15: -29.5548267, -11.9284811, -29.5594158, -11.9128571, -12.1358070, 12.1158028
16: -43.3819580, -22.5598869, -43.3667145, -22.5502720, -16.2324295, 16.2125778
17: -99.9815598, -70.0290451, -99.9543839, -70.0134659, -22.1065826, 22.1569138
18: -17.7310467, 3.4444237, -17.7443733, 3.4597125, -13.6616364, 13.6880951
19: -21.0156384, -6.4605970, -21.0188332, -6.4506807, -12.3848152, 12.4161415
20: -8.1845703, 5.5857067, -8.1903954, 5.5871620, -13.7717323, 13.7761021
21: -30.4718170, -12.1731987, -30.4794312, -12.1579542, -16.0537033, 16.1000290
22: -24.8058090, -8.3610096, -24.8042412, -8.3592091, -12.1348648, 12.1690750
23: -16.8599091, 0.1263595, -16.8771591, 0.1407021, -14.0920868, 14.1375275
24: -8.0102692, 6.8971286, -8.0143757, 6.8995719, -12.7503052, 12.7786674
25: -4.5684838, 11.7003441, -4.5918188, 11.7132883, -14.1254807, 14.1530037
26: -23.0320778, -1.5675237, -23.0457516, -1.5654547, -18.2697296, 18.3172073
27: -17.7984829, -3.7870960, -17.8045444, -3.7880526, -12.8815994, 12.9049568
28: -3.3153791, 16.1500053, -3.3317637, 16.1570835, -15.9439621, 15.9551392
29: -41.7307091, -23.3642025, -41.7347412, -23.3591576, -14.5270462, 14.5473557
30: -11.7765207, 7.2299852, -11.7943411, 7.2404003, -17.7012711, 17.7199173
31: -22.9036293, -4.3928766, -22.9082260, -4.3876114, -15.2245178, 15.2622719
32: -3.7638645, 10.5780773, -3.7798502, 10.5997400, -11.2233086, 11.2074471
33: 10.5263863, 30.8776131, 10.5018597, 30.8699799, -16.2734528, 16.2912521
34: 11.2739544, 29.0112057, 11.2671165, 28.9861927, -11.4029312, 11.4362068
35: 22.9482899, 40.4804535, 22.9121437, 40.4708824, -11.2768860, 11.3196220
36: 17.9430046, 34.5429230, 17.9388924, 34.5287552, -12.3558197, 12.3754463
37: 7.9075627, 28.0753441, 7.8676996, 28.0957832, -16.7265930, 16.7380219
38: 6.5933199, 26.6052361, 6.5846276, 26.5953693, -14.4015732, 14.4113617
39: 5.6993318, 25.9501419, 5.6973982, 25.9490013, -16.2263794, 16.2213364
40: 0.6156330, 19.8632774, 0.6024227, 19.8662415, -12.5908737, 12.5947342
41: -4.0871348, 9.0983868, -4.0895424, 9.0891342, -10.9557343, 10.9588127
42: -27.5808353, -10.8440580, -27.5847359, -10.8482513, -11.5139198, 11.5351486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 754

## Relational analysis of IS_B2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1950490, upper bound: 5.1937625
time: 6.22 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2142800, upper bound: 5.1944269
time: 5.96 seconds

## BFS IS instance: IS_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -57.6169968, -32.6752930, -57.6386948, -32.6483078, -17.4380112, 17.4682846
1: -39.1717949, -20.2501678, -39.1867523, -20.2269173, -11.8205948, 11.8390198
2: -27.2175522, -11.1917944, -27.2293663, -11.1755867, -10.9715424, 10.9818268
3: -31.5690956, -14.0990982, -31.5732002, -14.0899429, -10.9337959, 10.9283333
4: -29.3721962, -8.6749554, -29.3890038, -8.6547327, -14.1178856, 14.1590462
5: -31.7395344, -13.5685616, -31.7462597, -13.5535889, -12.1429214, 12.1339874
6: -14.8431997, 2.8710136, -14.8679142, 2.8853827, -11.5742149, 11.5536919
7: -46.6252174, -25.5875111, -46.6403503, -25.5627861, -11.9575920, 11.9748421
8: -41.4155388, -19.9059830, -41.4298706, -19.8733616, -10.6213760, 10.6460705
9: -24.2658195, -5.1564531, -24.2778740, -5.1437378, -16.4349365, 16.4456406
10: -52.0707703, -29.7194061, -52.0945854, -29.6854362, -16.9775581, 17.0756531
11: -47.9058113, -27.1024857, -47.9093895, -27.0947037, -15.0761490, 15.0498428
12: -13.3493614, 5.9013104, -13.3524246, 5.9109845, -15.3714600, 15.3981857
13: -9.2377453, 9.7076569, -9.2599106, 9.7293930, -16.2359390, 16.2649994
14: -86.0429001, -59.5969162, -86.0858612, -59.5481682, -19.7931366, 19.8320656
15: -29.5524445, -11.9535332, -29.5659561, -11.9356918, -12.1032181, 12.1100426
16: -43.3564606, -22.6028404, -43.3721657, -22.5800629, -16.1848679, 16.1923828
17: -99.9417267, -70.1158371, -99.9701309, -70.0726852, -22.1311264, 22.0632095
18: -17.7466030, 3.4584608, -17.7528305, 3.4613402, -13.6930008, 13.6541138
19: -21.0063610, -6.4551868, -21.0116653, -6.4476161, -12.4067574, 12.4046326
20: -8.1687794, 5.5779877, -8.1809969, 5.5838842, -13.7526636, 13.7589846
21: -30.4705372, -12.1561060, -30.4759789, -12.1485052, -16.0945435, 16.0736389
22: -24.7939816, -8.3656340, -24.7983112, -8.3617897, -12.1480141, 12.1273880
23: -16.8660507, 0.1425545, -16.8721733, 0.1517590, -14.1276550, 14.1116524
24: -7.9893708, 6.8933916, -8.0010357, 6.9000916, -12.7615738, 12.7417336
25: -4.5702114, 11.7208014, -4.5807400, 11.7265234, -14.1508026, 14.1435814
26: -23.0308037, -1.5761490, -23.0437336, -1.5695348, -18.2932816, 18.2323761
27: -17.7835598, -3.7958195, -17.7953281, -3.7898235, -12.8834991, 12.8559494
28: -3.2993217, 16.1566200, -3.3141243, 16.1664162, -15.9387360, 15.9420776
29: -41.7272415, -23.3641014, -41.7314148, -23.3579178, -14.5319824, 14.5198746
30: -11.7723732, 7.2382040, -11.7832994, 7.2512226, -17.7106628, 17.6978760
31: -22.8821602, -4.3984375, -22.8944397, -4.3911514, -15.2530365, 15.2360687
32: -3.7636337, 10.5970802, -3.7799606, 10.6007233, -11.1972313, 11.2311859
33: 10.5922623, 30.8663845, 10.5529461, 30.8847294, -16.2204971, 16.2116013
34: 11.3447876, 28.9664803, 11.3098011, 28.9860229, -11.3589401, 11.3534851
35: 23.0010223, 40.4718819, 22.9635944, 40.4914856, -11.2745514, 11.2105217
36: 18.0269337, 34.5180168, 17.9902763, 34.5309143, -12.3043518, 12.3018074
37: 7.8974562, 28.1246471, 7.8837905, 28.1293354, -16.7355042, 16.7145844
38: 6.6692600, 26.5871773, 6.6317520, 26.5944843, -14.3321228, 14.3512497
39: 5.7728524, 25.9443741, 5.7401590, 25.9479542, -16.1481094, 16.1829224
40: 0.6400237, 19.8513336, 0.6176133, 19.8635483, -12.5480461, 12.5783920
41: -4.0585041, 9.0803871, -4.0724392, 9.0892630, -10.9438324, 10.9550972
42: -27.5643616, -10.8613873, -27.5741444, -10.8517847, -11.5138474, 11.5113449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 754

## Relational analysis of IS_B2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1730533, upper bound: 5.1932042
time: 6.90 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1738806, upper bound: 5.2125341
time: 14.23 seconds

## BFS IS instance: IS_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -57.6401787, -32.6404800, -57.6431084, -32.6277657, -17.4816284, 17.4874611
1: -39.1868973, -20.2186832, -39.1879349, -20.2082939, -11.8533173, 11.8529587
2: -27.2294502, -11.1696062, -27.2303772, -11.1626806, -10.9963455, 10.9934196
3: -31.5710907, -14.0862598, -31.5733089, -14.0822725, -10.9454880, 10.9426804
4: -29.3884506, -8.6458015, -29.3896599, -8.6376581, -14.1506920, 14.1675262
5: -31.7460594, -13.5500526, -31.7468109, -13.5420866, -12.1602707, 12.1498375
6: -14.8705549, 2.8855519, -14.8839188, 2.8862047, -11.5684929, 11.5842400
7: -46.6401062, -25.5579834, -46.6408653, -25.5449753, -11.9902687, 11.9846535
8: -41.4298515, -19.8662720, -41.4303589, -19.8496857, -10.6581955, 10.6614017
9: -24.2770519, -5.1380730, -24.2784195, -5.1327772, -16.4575653, 16.4516144
10: -52.0938644, -29.6772079, -52.0949097, -29.6606827, -17.0267601, 17.0856705
11: -47.9103966, -27.1012230, -47.9112587, -27.0937271, -15.0775070, 15.0778618
12: -13.3519392, 5.9087696, -13.3564157, 5.9132004, -15.3880615, 15.4195900
13: -9.2603388, 9.7376757, -9.2679462, 9.7465830, -16.2759476, 16.2984161
14: -86.0846176, -59.5394859, -86.0885315, -59.5138741, -19.8670120, 19.8315468
15: -29.5657272, -11.9270849, -29.5679607, -11.9202747, -12.1317177, 12.1263504
16: -43.3726883, -22.5775299, -43.3748169, -22.5646439, -16.2115860, 16.2089348
17: -99.9688721, -70.0662079, -99.9712524, -70.0430298, -22.1251373, 22.1131592
18: -17.7527008, 3.4607558, -17.7558098, 3.4621701, -13.6977310, 13.6734390
19: -21.0141449, -6.4565172, -21.0168266, -6.4477820, -12.4115295, 12.4254761
20: -8.1858130, 5.5813093, -8.1898804, 5.5851045, -13.7709179, 13.7711897
21: -30.4778709, -12.1577826, -30.4797859, -12.1490192, -16.0989532, 16.1076012
22: -24.7995758, -8.3675957, -24.8025436, -8.3629093, -12.1516266, 12.1580811
23: -16.8743439, 0.1436371, -16.8763199, 0.1523322, -14.1335220, 14.1408043
24: -8.0089436, 6.8984566, -8.0124035, 6.9006448, -12.7792053, 12.7736206
25: -4.5854344, 11.7231140, -4.5889745, 11.7268581, -14.1642609, 14.1689110
26: -23.0480423, -1.5680110, -23.0528889, -1.5660253, -18.3085632, 18.2836456
27: -17.7984924, -3.7903702, -17.8032284, -3.7884457, -12.8970795, 12.8834267
28: -3.3232949, 16.1665459, -3.3277490, 16.1682720, -15.9635086, 15.9649963
29: -41.7324066, -23.3631859, -41.7341347, -23.3576355, -14.5379486, 14.5422897
30: -11.7927151, 7.2501860, -11.7943916, 7.2525611, -17.7336578, 17.7311554
31: -22.8984795, -4.4010129, -22.9036751, -4.3923130, -15.2683945, 15.2545052
32: -3.7811430, 10.6002598, -3.7901015, 10.6012030, -11.2139053, 11.2447777
33: 10.5431824, 30.8842926, 10.5242176, 30.8853378, -16.2459946, 16.2557602
34: 11.3062811, 28.9862404, 11.2867432, 28.9867554, -11.3791351, 11.3944473
35: 22.9548588, 40.4908447, 22.9356995, 40.4915390, -11.2948799, 11.2566528
36: 17.9843063, 34.5306625, 17.9644623, 34.5311241, -12.3256645, 12.3391037
37: 7.8766618, 28.1303730, 7.8713427, 28.1296425, -16.7479553, 16.7312927
38: 6.6305895, 26.5949059, 6.6080523, 26.5957508, -14.3693542, 14.3788071
39: 5.7343392, 25.9471626, 5.7175703, 25.9484310, -16.1866226, 16.2084351
40: 0.6133852, 19.8640118, 0.6017566, 19.8652401, -12.5696869, 12.5951843
41: -4.0745673, 9.0893269, -4.0816345, 9.0901833, -10.9469261, 10.9722176
42: -27.5771904, -10.8516483, -27.5814838, -10.8498507, -11.5163956, 11.5283585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 754

## Relational analysis of IS_B2_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1899101, upper bound: 5.1746529
time: 205.93 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1905843, upper bound: 5.2146814
time: 12.76 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -57.6450081, -32.6461334, -57.6418076, -32.6316986, -17.4848518, 17.4934158
1: -39.1890182, -20.2266960, -39.1875534, -20.2136116, -11.8522491, 11.8567238
2: -27.2291069, -11.1762667, -27.2304001, -11.1670971, -10.9924507, 10.9953842
3: -31.5726013, -14.0900612, -31.5735970, -14.0851841, -10.9452553, 10.9406166
4: -29.3857079, -8.6562366, -29.3899269, -8.6444664, -14.1428146, 14.1730423
5: -31.7479973, -13.5519543, -31.7470322, -13.5439987, -12.1619034, 12.1514320
6: -14.8652821, 2.8889999, -14.8802471, 2.8860974, -11.5865707, 11.5869370
7: -46.6489449, -25.5578766, -46.6410522, -25.5458641, -11.9982872, 11.9972572
8: -41.4359245, -19.8671684, -41.4302597, -19.8512764, -10.6640625, 10.6771584
9: -24.2781715, -5.1431632, -24.2787704, -5.1366816, -16.4546890, 16.4550629
10: -52.0993118, -29.6794415, -52.0947952, -29.6637650, -17.0309296, 17.0977097
11: -47.9192047, -27.0902920, -47.9103241, -27.0881310, -15.0710602, 15.0620308
12: -13.3584785, 5.9103365, -13.3570175, 5.9130926, -15.3895531, 15.4188919
13: -9.2632160, 9.7248421, -9.2697201, 9.7388763, -16.2718277, 16.2913742
14: -86.1076202, -59.5377808, -86.0885010, -59.5137939, -19.8934784, 19.8689995
15: -29.5608521, -11.9376221, -29.5682392, -11.9269886, -12.1212883, 12.1268768
16: -43.3843613, -22.5750504, -43.3742294, -22.5643520, -16.2163391, 16.2144012
17: -99.9881134, -70.0629425, -99.9714355, -70.0426025, -22.1772766, 22.0939331
18: -17.7517948, 3.4585629, -17.7556858, 3.4594476, -13.6959991, 13.6646919
19: -21.0209427, -6.4438071, -21.0149956, -6.4414382, -12.4124794, 12.4138222
20: -8.1780081, 5.5861249, -8.1847801, 5.5877790, -13.7657871, 13.7709045
21: -30.4885178, -12.1426420, -30.4782410, -12.1411142, -16.0986481, 16.0867424
22: -24.8055954, -8.3583565, -24.8016548, -8.3581181, -12.1562119, 12.1385498
23: -16.8810997, 0.1590608, -16.8745308, 0.1604083, -14.1391525, 14.1246567
24: -7.9974432, 6.8972321, -8.0051184, 6.9012227, -12.7695389, 12.7553978
25: -4.5827475, 11.7291775, -4.5850315, 11.7308378, -14.1614990, 14.1538811
26: -23.0412560, -1.5682213, -23.0486488, -1.5680299, -18.3032455, 18.2618179
27: -17.7898674, -3.7917421, -17.7984886, -3.7886302, -12.8893280, 12.8679276
28: -3.3109596, 16.1596699, -3.3194211, 16.1666241, -15.9527969, 15.9528503
29: -41.7359772, -23.3539352, -41.7332382, -23.3530521, -14.5360031, 14.5296249
30: -11.7778635, 7.2433872, -11.7851763, 7.2531385, -17.7206802, 17.7117386
31: -22.8962593, -4.3907871, -22.9010773, -4.3872766, -15.2642975, 15.2484283
32: -3.7826824, 10.6045198, -3.7906976, 10.6012840, -11.2132339, 11.2468834
33: 10.5477839, 30.8877754, 10.5279675, 30.8853951, -16.2552032, 16.2587204
34: 11.3078079, 28.9931202, 11.2887783, 28.9866753, -11.3877983, 11.4012032
35: 22.9563618, 40.4964333, 22.9383297, 40.4914474, -11.3079720, 11.2601624
36: 17.9811401, 34.5348587, 17.9643917, 34.5310059, -12.3403091, 12.3433418
37: 7.8853369, 28.1237488, 7.8778300, 28.1273518, -16.7479706, 16.7265091
38: 6.6242990, 26.6009636, 6.6066380, 26.5953770, -14.3741608, 14.3885193
39: 5.7327075, 25.9469681, 5.7181096, 25.9474277, -16.1899796, 16.2120056
40: 0.6157422, 19.8700676, 0.6039171, 19.8651218, -12.5650291, 12.6131554
41: -4.0716238, 9.0912561, -4.0798082, 9.0901756, -10.9557114, 10.9737320
42: -27.5702629, -10.8513899, -27.5774975, -10.8490915, -11.5202026, 11.5244522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 754

## Relational analysis of IS_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1898328, upper bound: 5.1941874
time: 5.54 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1904856, upper bound: 5.2132882
time: 5.71 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -57.6682434, -32.6113701, -57.6462250, -32.6111374, -17.5285034, 17.5125313
1: -39.2041168, -20.1952438, -39.1887512, -20.1949463, -11.8849907, 11.8706627
2: -27.2410049, -11.1540680, -27.2314262, -11.1541290, -11.0172691, 11.0070000
3: -31.5745983, -14.0771961, -31.5737457, -14.0775137, -10.9571228, 10.9552040
4: -29.4019566, -8.6270218, -29.3906136, -8.6273336, -14.1756325, 14.1815643
5: -31.7545223, -13.5334167, -31.7475605, -13.5325165, -12.1792564, 12.1673203
6: -14.8926239, 2.9035435, -14.8962564, 2.8868651, -11.5808296, 11.6174660
7: -46.6638222, -25.5283165, -46.6415977, -25.5279922, -12.0309448, 12.0071220
8: -41.4501686, -19.8274250, -41.4307747, -19.8275986, -10.7008743, 10.6925259
9: -24.2894382, -5.1246967, -24.2793217, -5.1256976, -16.4773102, 16.4610901
10: -52.1224174, -29.6371708, -52.0951614, -29.6389866, -17.0801849, 17.1077881
11: -47.9238281, -27.0889587, -47.9121895, -27.0871735, -15.0729141, 15.0900650
12: -13.3611603, 5.9181633, -13.3610411, 5.9152775, -15.4061852, 15.4407425
13: -9.2858181, 9.7548428, -9.2777395, 9.7560596, -16.3118439, 16.3247948
14: -86.1494141, -59.4802856, -86.0912247, -59.4794617, -19.9674149, 19.8685112
15: -29.5741844, -11.9111471, -29.5702477, -11.9115515, -12.1497688, 12.1431885
16: -43.4006310, -22.5497055, -43.3768654, -22.5489941, -16.2467422, 16.2312393
17: -100.0159912, -70.0131073, -99.9725494, -70.0128860, -22.1807709, 22.1444397
18: -17.7580070, 3.4608345, -17.7586498, 3.4602757, -13.7007751, 13.6841698
19: -21.0289497, -6.4444914, -21.0201607, -6.4414797, -12.4182396, 12.4346619
20: -8.1950159, 5.5893536, -8.1936407, 5.5889969, -13.7840128, 13.7829943
21: -30.4961319, -12.1438599, -30.4820175, -12.1415634, -16.1045227, 16.1205711
22: -24.8114567, -8.3599682, -24.8058662, -8.3590069, -12.1605225, 12.1691628
23: -16.8894234, 0.1602116, -16.8786602, 0.1609889, -14.1457977, 14.1531830
24: -8.0169830, 6.9021444, -8.0165148, 6.9017916, -12.7871017, 12.7873268
25: -4.5972986, 11.7313147, -4.5932722, 11.7311487, -14.1753311, 14.1786575
26: -23.0585270, -1.5600863, -23.0577812, -1.5645456, -18.3185272, 18.3131409
27: -17.8047791, -3.7862897, -17.8063641, -3.7872667, -12.9028931, 12.8953705
28: -3.3349867, 16.1696014, -3.3330481, 16.1684895, -15.9775772, 15.9757462
29: -41.7412338, -23.3528671, -41.7359314, -23.3526802, -14.5424118, 14.5520859
30: -11.7982559, 7.2552018, -11.7963009, 7.2545280, -17.7436752, 17.7450714
31: -22.9125881, -4.3928442, -22.9102783, -4.3881512, -15.2799454, 15.2670860
32: -3.8002434, 10.6077547, -3.8008289, 10.6017761, -11.2300644, 11.2606964
33: 10.4986649, 30.9057274, 10.4992504, 30.8860397, -16.2807922, 16.3028946
34: 11.2693043, 29.0128727, 11.2656670, 28.9874172, -11.4080200, 11.4421616
35: 22.9101143, 40.5153961, 22.9104233, 40.4915085, -11.3283882, 11.3062859
36: 17.9384212, 34.5475159, 17.9385719, 34.5312119, -12.3617401, 12.3806419
37: 7.8644509, 28.1295013, 7.8653431, 28.1277199, -16.7612762, 16.7430153
38: 6.5854406, 26.6087399, 6.5829191, 26.5966835, -14.4116631, 14.4168091
39: 5.6940827, 25.9497604, 5.6955051, 25.9479599, -16.2286224, 16.2375259
40: 0.5889959, 19.8827477, 0.5880251, 19.8668365, -12.5873108, 12.6308517
41: -4.0877070, 9.1001873, -4.0890179, 9.0911160, -10.9588051, 10.9908409
42: -27.5830822, -10.8416691, -27.5848160, -10.8471375, -11.5227318, 11.5415001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 754

## Relational analysis of IS_B2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1959948, upper bound: 5.2145604
time: 5.67 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2152089, upper bound: 5.2152090
time: 5.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.46 seconds
IS_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1938873, upper bound: 5.1740077
IS_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.2132264, upper bound: 5.1746619
IS_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1945098, upper bound: 5.1934169
IS_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.2137963, upper bound: 5.1939131
IS_B2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1950490, upper bound: 5.1937625
IS_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.2142800, upper bound: 5.1944269
IS_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1730533, upper bound: 5.1932042
IS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1738806, upper bound: 5.2125341
IS_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1899101, upper bound: 5.1746529
IS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1905843, upper bound: 5.2146814
IS_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1898328, upper bound: 5.1941874
IS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1904856, upper bound: 5.2132882
IS_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.1959948, upper bound: 5.2145604
IS_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.46
Output dim: 35, lower bound: -5.2152089, upper bound: 5.2152090

## BFS IS instance: IS_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -57.6566811, -32.6596794, -57.6157837, -32.6929703, -17.4678650, 17.4308777
1: -39.2015915, -20.2300911, -39.1721344, -20.2550163, -11.8489265, 11.8165894
2: -27.2419815, -11.1776009, -27.2195129, -11.1958466, -10.9891853, 10.9667168
3: -31.5715866, -14.0897465, -31.5631256, -14.0993328, -10.9249802, 10.9246788
4: -29.4070854, -8.6550274, -29.3751221, -8.6761799, -14.1756783, 14.1110611
5: -31.7511749, -13.5596390, -31.7385883, -13.5781631, -12.1292305, 12.1325912
6: -14.8616428, 2.9037166, -14.8315010, 2.8697419, -11.5376587, 11.5790939
7: -46.6576385, -25.5758743, -46.6205978, -25.6091862, -11.9705467, 11.9430580
8: -41.4508209, -19.8920193, -41.4120750, -19.9386005, -10.6338806, 10.6011467
9: -24.2896633, -5.1459041, -24.2668190, -5.1615534, -16.4532013, 16.4262390
10: -52.1232910, -29.7027340, -52.0708809, -29.7528706, -17.0717506, 16.9729767
11: -47.9128761, -27.1147232, -47.8943710, -27.1347179, -15.0295486, 15.0909081
12: -13.3439636, 5.9093552, -13.3351002, 5.8962722, -15.3908997, 15.3590965
13: -9.2600727, 9.7336617, -9.2228279, 9.7120180, -16.2691193, 16.2195244
14: -86.1337585, -59.5757828, -86.0337067, -59.6429825, -19.8328629, 19.7569427
15: -29.5745926, -11.9342766, -29.5552120, -11.9513416, -12.1199226, 12.0964622
16: -43.3829803, -22.5944557, -43.3550758, -22.6247597, -16.1761093, 16.1712570
17: -99.9974289, -70.0910187, -99.9332581, -70.1473923, -22.0061188, 22.1260147
18: -17.7530861, 3.4602880, -17.7447014, 3.4530804, -13.6430473, 13.6958427
19: -21.0119858, -6.4739404, -20.9949379, -6.4920139, -12.3752289, 12.4130478
20: -8.1852360, 5.5721521, -8.1726799, 5.5583706, -13.7436066, 13.7448320
21: -30.4796143, -12.1768837, -30.4555569, -12.1985312, -16.0418701, 16.1124725
22: -24.7982464, -8.3694801, -24.7857151, -8.3767233, -12.1113930, 12.1576881
23: -16.8738251, 0.1265368, -16.8595486, 0.1041192, -14.0778427, 14.1413956
24: -8.0060682, 6.9017582, -7.9964876, 6.8899183, -12.7399902, 12.7791443
25: -4.5823212, 11.7171936, -4.5674152, 11.7029715, -14.1299210, 14.1586685
26: -23.0468407, -1.5705404, -23.0324478, -1.5869288, -18.2112732, 18.3059006
27: -17.7995453, -3.7874436, -17.7870007, -3.7999098, -12.8500595, 12.8960495
28: -3.3201809, 16.1720924, -3.3059020, 16.1558075, -15.9443512, 15.9494781
29: -41.7321014, -23.3664150, -41.7225647, -23.3791008, -14.5101242, 14.5418816
30: -11.7938375, 7.2609425, -11.7851686, 7.2417097, -17.7059708, 17.7364197
31: -22.8940029, -4.4170074, -22.8808441, -4.4349642, -15.2044449, 15.2493629
32: -3.7715766, 10.6022797, -3.7478952, 10.5925922, -11.2228432, 11.1850548
33: 10.5690022, 30.9048729, 10.6204376, 30.8683586, -16.1980743, 16.2135239
34: 11.3235178, 29.0093517, 11.3706961, 28.9621658, -11.3353004, 11.3564415
35: 22.9838810, 40.5165596, 23.0374432, 40.4708023, -11.1929970, 11.2632675
36: 18.0147934, 34.5476456, 18.0700073, 34.5151634, -12.2785263, 12.2801704
37: 7.8803787, 28.1349773, 7.8912578, 28.1242218, -16.7017899, 16.7393646
38: 6.6529679, 26.6004868, 6.7071404, 26.5830250, -14.3263931, 14.2945404
39: 5.7561564, 25.9493790, 5.8019829, 25.9415665, -16.1595383, 16.1149673
40: 0.6206861, 19.8751011, 0.6448164, 19.8497581, -12.5662651, 12.5472031
41: -4.0695024, 9.0996752, -4.0520015, 9.0774918, -10.9408112, 10.9431725
42: -27.5785408, -10.8465347, -27.5713844, -10.8663216, -11.4983597, 11.5267296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 695

## Relational analysis of IS_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_B1_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2053408, upper bound: 5.1740398
time: 10.91 seconds

## Relational analysis of IS_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2126050, upper bound: 5.1740398
time: 5.69 seconds

## BFS IS instance: IS_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -57.6597824, -32.6433411, -57.6423149, -32.6656837, -17.4924164, 17.4773254
1: -39.2023659, -20.2169857, -39.1890411, -20.2321396, -11.8661308, 11.8477402
2: -27.2429981, -11.1694174, -27.2308712, -11.1813240, -11.0024719, 10.9872055
3: -31.5717373, -14.0851316, -31.5659580, -14.0907001, -10.9369049, 10.9358482
4: -29.4079819, -8.6449099, -29.3876152, -8.6578817, -14.1894455, 14.1355896
5: -31.7519493, -13.5501766, -31.7470131, -13.5619316, -12.1462784, 12.1513824
6: -14.8737917, 2.9044256, -14.8531895, 2.8876147, -11.5706253, 11.5910835
7: -46.6583405, -25.5591011, -46.6441650, -25.5800934, -11.9924660, 11.9834290
8: -41.4512482, -19.8701782, -41.4322281, -19.9004555, -10.6642990, 10.6434269
9: -24.2905731, -5.1394320, -24.2788906, -5.1491284, -16.4621506, 16.4465637
10: -52.1234512, -29.6811180, -52.0992584, -29.7132435, -17.0934372, 17.0260086
11: -47.9137306, -27.1082039, -47.9074249, -27.1228523, -15.0414124, 15.0855560
12: -13.3485193, 5.9111562, -13.3438406, 5.9042406, -15.4111328, 15.3763962
13: -9.2697639, 9.7425032, -9.2477684, 9.7283268, -16.2945633, 16.2546310
14: -86.1364059, -59.5415802, -86.0980606, -59.5842743, -19.8693008, 19.8566704
15: -29.5768375, -11.9261408, -29.5634918, -11.9366283, -12.1362267, 12.1141434
16: -43.3850327, -22.5796909, -43.3802948, -22.5995407, -16.1977386, 16.2044716
17: -99.9986496, -70.0611038, -99.9797287, -70.0948715, -22.0389557, 22.1681900
18: -17.7558899, 3.4583893, -17.7496357, 3.4529595, -13.6533127, 13.6985168
19: -21.0152817, -6.4681363, -21.0088825, -6.4813147, -12.3837090, 12.4175034
20: -8.1889448, 5.5759621, -8.1814814, 5.5657640, -13.7547092, 13.7574434
21: -30.4818192, -12.1694574, -30.4735355, -12.1854763, -16.0532227, 16.1153831
22: -24.8015327, -8.3665543, -24.7972069, -8.3702698, -12.1227226, 12.1648903
23: -16.8761044, 0.1350464, -16.8724060, 0.1203437, -14.0905685, 14.1504784
24: -8.0101204, 6.9027185, -8.0042715, 6.8933740, -12.7534409, 12.7868233
25: -4.5865836, 11.7203350, -4.5791259, 11.7097969, -14.1396103, 14.1683960
26: -23.0516205, -1.5690155, -23.0425186, -1.5792525, -18.2399597, 18.3153381
27: -17.8026123, -3.7862735, -17.7930775, -3.7960176, -12.8617783, 12.9016914
28: -3.3253720, 16.1722813, -3.3172181, 16.1584015, -15.9541473, 15.9626617
29: -41.7339096, -23.3615398, -41.7304993, -23.3692360, -14.5192490, 14.5451927
30: -11.7955713, 7.2629213, -11.7904425, 7.2465954, -17.7194061, 17.7461014
31: -22.9005013, -4.4129272, -22.8932762, -4.4273286, -15.2163467, 15.2599678
32: -3.7821381, 10.6028328, -3.7664509, 10.5998840, -11.2382469, 11.2006721
33: 10.5442276, 30.9055786, 10.5765829, 30.8897152, -16.2447739, 16.2475357
34: 11.3026142, 29.0100060, 11.3341732, 28.9887657, -11.3827820, 11.3848419
35: 22.9588013, 40.5165176, 22.9933128, 40.4953079, -11.2424469, 11.2961235
36: 17.9891205, 34.5477409, 18.0249062, 34.5319672, -12.3198013, 12.3154488
37: 7.8744898, 28.1329689, 7.8795595, 28.1231575, -16.7134094, 16.7513466
38: 6.6280489, 26.6013603, 6.6628304, 26.5966797, -14.3631134, 14.3358498
39: 5.7343698, 25.9489059, 5.7625813, 25.9441071, -16.1881104, 16.1560364
40: 0.6071339, 19.8766632, 0.6209412, 19.8683796, -12.5992813, 12.5636978
41: -4.0767889, 9.1006021, -4.0648928, 9.0882683, -10.9592743, 10.9548416
42: -27.5816689, -10.8436718, -27.5769215, -10.8565350, -11.5111237, 11.5329514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 695

## Relational analysis of IS_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_B1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2059115, upper bound: 5.1932912
time: 23.70 seconds

## Relational analysis of IS_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2131746, upper bound: 5.1932912
time: 20.94 seconds

## BFS IS instance: IS_B2_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -57.6702347, -32.6200943, -57.6368713, -32.6132545, -17.5326157, 17.4930267
1: -39.2094345, -20.2023544, -39.1830139, -20.1966515, -11.8892555, 11.8457527
2: -27.2472095, -11.1572924, -27.2273483, -11.1551523, -11.0258789, 10.9960060
3: -31.5728149, -14.0821075, -31.5714054, -14.0790958, -10.9563408, 10.9530869
4: -29.3935261, -8.6526718, -29.3741322, -8.6291752, -14.1916656, 14.1346703
5: -31.7577972, -13.5402632, -31.7470455, -13.5341301, -12.1810341, 12.1618500
6: -14.8860369, 2.9220295, -14.8917980, 2.8861260, -11.5649681, 11.6292648
7: -46.6724815, -25.5337048, -46.6363602, -25.5292778, -12.0415344, 11.9982681
8: -41.4625320, -19.8345261, -41.4256210, -19.8297043, -10.7194862, 10.6744080
9: -24.2752266, -5.1492825, -24.2633381, -5.1276736, -16.4607849, 16.4166183
10: -52.1056824, -29.6810837, -52.0681839, -29.6423531, -17.1103439, 17.0276184
11: -47.9189224, -27.1008186, -47.9100456, -27.0939598, -15.0242310, 15.0870781
12: -13.3226757, 5.8937068, -13.3410177, 5.9101944, -15.4045486, 15.3983231
13: -9.2926207, 9.7440014, -9.2760754, 9.7509460, -16.3298035, 16.2869797
14: -86.1344223, -59.5294266, -86.0538940, -59.4819298, -19.9522095, 19.8351707
15: -29.5656509, -11.9283400, -29.5591240, -11.9136505, -12.1455193, 12.1119537
16: -43.3939438, -22.5602417, -43.3663940, -22.5509949, -16.2402039, 16.2078362
17: -100.0115967, -70.0300674, -99.9533691, -70.0147247, -22.1017761, 22.1548996
18: -17.7336636, 3.4447277, -17.7439575, 3.4587398, -13.6556740, 13.6924210
19: -21.0170784, -6.4645824, -21.0185089, -6.4532857, -12.3785477, 12.4170647
20: -8.1865692, 5.5856390, -8.1896505, 5.5862827, -13.7728519, 13.7752895
21: -30.4759865, -12.1762714, -30.4791107, -12.1598721, -16.0424652, 16.1060791
22: -24.8079548, -8.3614731, -24.8037605, -8.3596935, -12.1276245, 12.1757622
23: -16.8619156, 0.1230227, -16.8768883, 0.1382796, -14.0817490, 14.1405296
24: -8.0113058, 6.9015999, -8.0136757, 6.8992753, -12.7426605, 12.7894859
25: -4.5698395, 11.7014809, -4.5910158, 11.7131796, -14.1215973, 14.1575584
26: -23.0343933, -1.5661068, -23.0451584, -1.5668471, -18.2537308, 18.3340759
27: -17.8009186, -3.7818427, -17.8040123, -3.7882919, -12.8750992, 12.9173393
28: -3.3173108, 16.1577225, -3.3309913, 16.1567459, -15.9452515, 15.9619522
29: -41.7317352, -23.3626060, -41.7337036, -23.3593159, -14.5245819, 14.5531502
30: -11.7785816, 7.2431569, -11.7935143, 7.2399173, -17.6978607, 17.7375412
31: -22.9054222, -4.3972683, -22.9076080, -4.3904819, -15.2207642, 15.2597084
32: -3.7640958, 10.5807447, -3.7788897, 10.5988350, -11.2156944, 11.2065239
33: 10.5263224, 30.8986740, 10.5035534, 30.8696423, -16.2672272, 16.3106537
34: 11.2731562, 29.0357494, 11.2683134, 28.9854927, -11.3982430, 11.4592018
35: 22.9482765, 40.5057678, 22.9137936, 40.4705238, -11.2693138, 11.3425980
36: 17.9435616, 34.5599670, 17.9403572, 34.5285797, -12.3495750, 12.3910751
37: 7.9075332, 28.0806904, 7.8685703, 28.0954704, -16.7254868, 16.7417564
38: 6.5937839, 26.6119251, 6.5859199, 26.5949173, -14.3975677, 14.4142456
39: 5.6988478, 25.9497795, 5.6987271, 25.9466019, -16.2250290, 16.2209549
40: 0.6162786, 19.8764038, 0.6032791, 19.8657017, -12.5828247, 12.6033096
41: -4.0877552, 9.1100569, -4.0888901, 9.0888062, -10.9513550, 10.9687538
42: -27.5810394, -10.8354435, -27.5842342, -10.8488045, -11.5098419, 11.5439339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 695

## Relational analysis of IS_B2_A1_A2_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2063938, upper bound: 5.1938078
time: 11.13 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2136589, upper bound: 5.1938078
time: 13.56 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -57.6166077, -32.6765404, -57.6578484, -32.6489182, -17.4310303, 17.4864502
1: -39.1714668, -20.2510490, -39.2022095, -20.2274857, -11.8148804, 11.8534546
2: -27.2173538, -11.1924973, -27.2429714, -11.1751900, -10.9682693, 10.9946404
3: -31.5678082, -14.0997086, -31.5740738, -14.0900373, -10.9330444, 10.9294395
4: -29.3717899, -8.6758614, -29.4087086, -8.6546535, -14.1118431, 14.1779137
5: -31.7394371, -13.5693607, -31.7520447, -13.5544090, -12.1419678, 12.1391373
6: -14.8421850, 2.8707228, -14.8679333, 2.9045234, -11.5935516, 11.5410080
7: -46.6250610, -25.5884743, -46.6586189, -25.5638714, -11.9511032, 11.9921074
8: -41.4154358, -19.9074936, -41.4512100, -19.8738766, -10.6121902, 10.6657276
9: -24.2656670, -5.1571555, -24.2909508, -5.1437988, -16.4304657, 16.4578476
10: -52.0706406, -29.7209244, -52.1234550, -29.6840401, -16.9682655, 17.1034470
11: -47.9056091, -27.1032448, -47.9128876, -27.0953217, -15.0853729, 15.0402298
12: -13.3484802, 5.8978977, -13.3490458, 5.9132872, -15.3728065, 15.3970222
13: -9.2368946, 9.7067719, -9.2711773, 9.7306042, -16.2351913, 16.2753487
14: -86.0422974, -59.5987167, -86.1363831, -59.5499725, -19.7727890, 19.8789711
15: -29.5520935, -11.9543304, -29.5767746, -11.9355316, -12.0993996, 12.1197548
16: -43.3561325, -22.6036110, -43.3841400, -22.5803719, -16.1803894, 16.2021370
17: -99.9405594, -70.1171646, -100.0000534, -70.0736771, -22.1293640, 22.0585251
18: -17.7461338, 3.4575462, -17.7554169, 3.4616253, -13.6975098, 13.6481857
19: -21.0060043, -6.4579430, -21.0127945, -6.4515977, -12.4082069, 12.3989372
20: -8.1680155, 5.5770564, -8.1830082, 5.5837250, -13.7517405, 13.7600651
21: -30.4702072, -12.1581497, -30.4801254, -12.1515779, -16.1022568, 16.0638199
22: -24.7935143, -8.3661251, -24.8002930, -8.3622818, -12.1549301, 12.1199646
23: -16.8657589, 0.1400726, -16.8738708, 0.1484356, -14.1331329, 14.1027451
24: -7.9886456, 6.8931046, -8.0020561, 6.9047441, -12.7730942, 12.7343597
25: -4.5692987, 11.7207203, -4.5820255, 11.7276840, -14.1560059, 14.1395454
26: -23.0302544, -1.5775120, -23.0461063, -1.5681295, -18.3101425, 18.2167130
27: -17.7830276, -3.7960479, -17.7978001, -3.7845669, -12.8958588, 12.8495598
28: -3.2985313, 16.1562786, -3.3160357, 16.1741505, -15.9455795, 15.9434128
29: -41.7262001, -23.3643131, -41.7325134, -23.3563004, -14.5378647, 14.5173607
30: -11.7714090, 7.2376986, -11.7852545, 7.2643642, -17.7284164, 17.6944580
31: -22.8815479, -4.4014692, -22.8960838, -4.3958793, -15.2504730, 15.2323303
32: -3.7626328, 10.5961189, -3.7802763, 10.6033764, -11.1963768, 11.2235641
33: 10.5939541, 30.8659821, 10.5529108, 30.9057732, -16.2399521, 16.2053833
34: 11.3459883, 28.9657879, 11.3089752, 29.0106430, -11.3819695, 11.3488235
35: 23.0026855, 40.4715271, 22.9635868, 40.5168610, -11.2975426, 11.2029343
36: 18.0284328, 34.5178375, 17.9908390, 34.5479965, -12.3199425, 12.2955551
37: 7.8984075, 28.1242943, 7.8837767, 28.1346512, -16.7392120, 16.7134857
38: 6.6705828, 26.5867119, 6.6322107, 26.6011868, -14.3349380, 14.3472443
39: 5.7741957, 25.9418087, 5.7396202, 25.9474678, -16.1478348, 16.1816406
40: 0.6408978, 19.8508339, 0.6182861, 19.8766899, -12.5571861, 12.5703163
41: -4.0578375, 9.0800476, -4.0731025, 9.1009617, -10.9537239, 10.9507332
42: -27.5638466, -10.8619328, -27.5743542, -10.8431482, -11.5226288, 11.5073242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1732593, upper bound: 5.2046499
time: 4.99 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1732593, upper bound: 5.2119125
time: 11.23 seconds

## BFS IS instance: IS_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -57.6398010, -32.6416779, -57.6622810, -32.6283684, -17.4746628, 17.5056190
1: -39.1865768, -20.2195511, -39.2034111, -20.2088604, -11.8476143, 11.8672676
2: -27.2292690, -11.1703043, -27.2439842, -11.1622562, -10.9930954, 11.0062370
3: -31.5698109, -14.0868702, -31.5742188, -14.0823202, -10.9447975, 10.9439087
4: -29.3880463, -8.6466875, -29.4094143, -8.6375589, -14.1446381, 14.1862183
5: -31.7459755, -13.5508280, -31.7525749, -13.5429087, -12.1593399, 12.1550293
6: -14.8695154, 2.8852706, -14.8839073, 2.9053230, -11.5877609, 11.5716362
7: -46.6399078, -25.5589294, -46.6591568, -25.5459976, -11.9837646, 12.0019341
8: -41.4296875, -19.8677444, -41.4517517, -19.8501949, -10.6490326, 10.6808605
9: -24.2769012, -5.1387501, -24.2914848, -5.1327820, -16.4530640, 16.4638367
10: -52.0937233, -29.6786671, -52.1237755, -29.6593208, -17.0174751, 17.1134567
11: -47.9102402, -27.1018639, -47.9146957, -27.0943298, -15.0867462, 15.0682106
12: -13.3511276, 5.9055777, -13.3530827, 5.9155326, -15.3894958, 15.4187431
13: -9.2594891, 9.7368059, -9.2791824, 9.7477961, -16.2752304, 16.3087311
14: -86.0840302, -59.5412369, -86.1391144, -59.5156174, -19.8467102, 19.8780365
15: -29.5653934, -11.9278679, -29.5787773, -11.9201365, -12.1279144, 12.1359940
16: -43.3723679, -22.5783100, -43.3867989, -22.5649586, -16.2070007, 16.2185707
17: -99.9677963, -70.0675201, -100.0012360, -70.0440369, -22.1200409, 22.1035233
18: -17.7522831, 3.4598286, -17.7584515, 3.4624600, -13.7022743, 13.6675644
19: -21.0137901, -6.4591126, -21.0180359, -6.4517550, -12.4129829, 12.4197731
20: -8.1850319, 5.5803814, -8.1918898, 5.5849757, -13.7700081, 13.7722712
21: -30.4775658, -12.1597004, -30.4839287, -12.1520786, -16.1066818, 16.0976639
22: -24.7991161, -8.3681040, -24.8045216, -8.3634167, -12.1583481, 12.1506310
23: -16.8741150, 0.1412235, -16.8780537, 0.1490003, -14.1385269, 14.1312523
24: -8.0082045, 6.8981810, -8.0134354, 6.9053097, -12.7906990, 12.7661476
25: -4.5846515, 11.7230320, -4.5903749, 11.7279787, -14.1695023, 14.1649132
26: -23.0475006, -1.5693934, -23.0552750, -1.5646150, -18.3252563, 18.2679825
27: -17.7979641, -3.7906225, -17.8056889, -3.7832031, -12.9094543, 12.8769760
28: -3.3225491, 16.1661797, -3.3296773, 16.1760254, -15.9703445, 15.9663162
29: -41.7313766, -23.3633575, -41.7353096, -23.3560123, -14.5438309, 14.5397644
30: -11.7918482, 7.2496777, -11.7964745, 7.2657204, -17.7513046, 17.7277451
31: -22.8978577, -4.4039860, -22.9053822, -4.3970528, -15.2658691, 15.2506943
32: -3.7802079, 10.5993271, -3.7903874, 10.6038752, -11.2130394, 11.2371635
33: 10.5448895, 30.8838863, 10.5241566, 30.9063835, -16.2652817, 16.2495651
34: 11.3074827, 28.9855270, 11.2859020, 29.0113678, -11.4019318, 11.3897552
35: 22.9565182, 40.4904785, 22.9356403, 40.5169067, -11.3179169, 11.2490921
36: 17.9857864, 34.5304832, 17.9650288, 34.5481834, -12.3412933, 12.3328705
37: 7.8775530, 28.1300411, 7.8712502, 28.1350632, -16.7515717, 16.7302513
38: 6.6318445, 26.5944595, 6.6084709, 26.6024666, -14.3715363, 14.3747635
39: 5.7356820, 25.9447746, 5.7170382, 25.9481087, -16.1863251, 16.2071304
40: 0.6142278, 19.8635178, 0.6024399, 19.8783836, -12.5751419, 12.5870399
41: -4.0739422, 9.0889759, -4.0822759, 9.1018658, -10.9567719, 10.9679070
42: -27.5766754, -10.8522186, -27.5816650, -10.8412056, -11.5251617, 11.5243263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_B2_A2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1899624, upper bound: 5.2067967
time: 20.04 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1899624, upper bound: 5.2140596
time: 7.56 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -57.6446075, -32.6473846, -57.6609383, -32.6323318, -17.4778595, 17.5115395
1: -39.1886826, -20.2276268, -39.2030411, -20.2141724, -11.8465233, 11.8711624
2: -27.2289104, -11.1769533, -27.2440319, -11.1666718, -10.9891777, 11.0081902
3: -31.5713005, -14.0906487, -31.5744915, -14.0852957, -10.9445877, 10.9418182
4: -29.3852730, -8.6571188, -29.4096336, -8.6443520, -14.1367798, 14.1919174
5: -31.7478676, -13.5527391, -31.7528114, -13.5448494, -12.1609573, 12.1565781
6: -14.8642254, 2.8887558, -14.8802319, 2.9052114, -11.6058922, 11.5742722
7: -46.6487236, -25.5588589, -46.6593704, -25.5468807, -11.9917831, 12.0145531
8: -41.4357452, -19.8686104, -41.4516373, -19.8518124, -10.6548767, 10.6968117
9: -24.2780075, -5.1438756, -24.2918243, -5.1366749, -16.4501877, 16.4672546
10: -52.0991783, -29.6809120, -52.1236725, -29.6623840, -17.0216179, 17.1254959
11: -47.9190598, -27.0909920, -47.9137802, -27.0887260, -15.0798645, 15.0524101
12: -13.3575850, 5.9071288, -13.3536224, 5.9153557, -15.3908615, 15.4179573
13: -9.2623739, 9.7239380, -9.2809315, 9.7400694, -16.2710648, 16.3016891
14: -86.1070404, -59.5395813, -86.1390381, -59.5155869, -19.8731079, 19.9159203
15: -29.5605507, -11.9384451, -29.5790157, -11.9268055, -12.1174507, 12.1365662
16: -43.3840179, -22.5758209, -43.3861694, -22.5647221, -16.2117538, 16.2241440
17: -99.9870148, -70.0643082, -100.0013733, -70.0435028, -22.1751556, 22.0891647
18: -17.7513390, 3.4576354, -17.7582836, 3.4597297, -13.7005043, 13.6587906
19: -21.0205765, -6.4463997, -21.0161819, -6.4454026, -12.4136200, 12.4082718
20: -8.1772175, 5.5852346, -8.1868315, 5.5876803, -13.7648983, 13.7720661
21: -30.4881878, -12.1445923, -30.4823704, -12.1441860, -16.1055527, 16.0772247
22: -24.8050842, -8.3588390, -24.8036537, -8.3585644, -12.1626892, 12.1313019
23: -16.8808327, 0.1566294, -16.8762531, 0.1571040, -14.1435394, 14.1157036
24: -7.9966965, 6.8969493, -8.0061855, 6.9057331, -12.7810059, 12.7478943
25: -4.5818520, 11.7290945, -4.5863295, 11.7319679, -14.1664734, 14.1499901
26: -23.0406914, -1.5695810, -23.0509529, -1.5666504, -18.3200531, 18.2461624
27: -17.7892990, -3.7919931, -17.8009357, -3.7834132, -12.9017105, 12.8614845
28: -3.3101916, 16.1593304, -3.3213613, 16.1743736, -15.9596405, 15.9541626
29: -41.7348709, -23.3540974, -41.7343903, -23.3514748, -14.5416565, 14.5270920
30: -11.7769585, 7.2428818, -11.7871418, 7.2663074, -17.7383957, 17.7083054
31: -22.8956776, -4.3936396, -22.9027481, -4.3917189, -15.2617188, 15.2448082
32: -3.7816758, 10.6035795, -3.7909970, 10.6038942, -11.2123528, 11.2392807
33: 10.5495138, 30.8873882, 10.5279789, 30.9064541, -16.2746048, 16.2524872
34: 11.3090258, 28.9924183, 11.2879810, 29.0113163, -11.4108009, 11.3965111
35: 22.9580383, 40.4960251, 22.9383049, 40.5168228, -11.3309517, 11.2525940
36: 17.9826355, 34.5346832, 17.9649334, 34.5480995, -12.3559074, 12.3370857
37: 7.8862686, 28.1233864, 7.8778086, 28.1326790, -16.7517090, 16.7254257
38: 6.6256237, 26.6005135, 6.6070628, 26.6020889, -14.3769913, 14.3844910
39: 5.7340717, 25.9443970, 5.7175760, 25.9469719, -16.1896667, 16.2106934
40: 0.6165962, 19.8695316, 0.6045842, 19.8782845, -12.5741692, 12.6050911
41: -4.0709610, 9.0909081, -4.0804486, 9.1018848, -10.9655914, 10.9693642
42: -27.5697250, -10.8519773, -27.5777092, -10.8404284, -11.5289612, 11.5204506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=113, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 695

## Relational analysis of IS_B2_A2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1898637, upper bound: 5.2054040
time: 5.65 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1898637, upper bound: 5.2126665
time: 5.55 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -57.6622009, -32.6352997, -57.6428261, -32.6246872, -17.5086594, 17.4852600
1: -39.2021370, -20.2159157, -39.1876297, -20.2066727, -11.8713493, 11.8488426
2: -27.2394943, -11.1696110, -27.2305450, -11.1629238, -11.0067520, 10.9903679
3: -31.5734959, -14.0859861, -31.5730705, -14.0825024, -10.9487038, 10.9442863
4: -29.4004288, -8.6468086, -29.3897133, -8.6385422, -14.1628647, 14.1606789
5: -31.7536888, -13.5463705, -31.7471085, -13.5398636, -12.1713142, 12.1541367
6: -14.8714657, 2.9021530, -14.8842640, 2.8860941, -11.5584831, 11.6040039
7: -46.6628151, -25.5493393, -46.6410446, -25.5399189, -12.0180359, 11.9854431
8: -41.4493523, -19.8591328, -41.4303131, -19.8455658, -10.6820221, 10.6603394
9: -24.2886925, -5.1373196, -24.2789021, -5.1328440, -16.4691315, 16.4473877
10: -52.1218414, -29.6677628, -52.0947990, -29.6563263, -17.0613861, 17.0755539
11: -47.9216194, -27.0911388, -47.9109001, -27.0883827, -15.0618744, 15.0817795
12: -13.3553085, 5.9124031, -13.3577433, 5.9117513, -15.3939362, 15.4265060
13: -9.2747517, 9.7349424, -9.2714634, 9.7447424, -16.2890854, 16.2981987
14: -86.1453094, -59.5221825, -86.0888443, -59.5031509, -19.9391174, 19.8231888
15: -29.5714722, -11.9281015, -29.5687046, -11.9211502, -12.1370659, 12.1241264
16: -43.3970642, -22.5662994, -43.3748245, -22.5584164, -16.2335815, 16.2127533
17: -100.0128098, -70.0433807, -99.9707947, -70.0300293, -22.1655807, 22.1273804
18: -17.7530365, 3.4579535, -17.7558632, 3.4586020, -13.6895676, 13.6753082
19: -21.0227451, -6.4467735, -21.0166550, -6.4427800, -12.4077606, 12.4276962
20: -8.1866655, 5.5871038, -8.1889238, 5.5877523, -13.7744179, 13.7760277
21: -30.4915276, -12.1452980, -30.4794502, -12.1423807, -16.0903473, 16.1108246
22: -24.8060169, -8.3604431, -24.8027458, -8.3593025, -12.1478729, 12.1612244
23: -16.8845520, 0.1576016, -16.8759041, 0.1594871, -14.1328964, 14.1438675
24: -8.0037479, 6.9011135, -8.0089855, 6.9012041, -12.7682228, 12.7752914
25: -4.5881419, 11.7309246, -4.5880499, 11.7309179, -14.1620483, 14.1703682
26: -23.0482712, -1.5648491, -23.0519657, -1.5672722, -18.2953949, 18.2963562
27: -17.7939758, -3.7883134, -17.8002396, -3.7884040, -12.8862686, 12.8844299
28: -3.3191895, 16.1667404, -3.3241260, 16.1668816, -15.9602432, 15.9639206
29: -41.7362289, -23.3543396, -41.7330246, -23.3535004, -14.5313339, 14.5448494
30: -11.7855816, 7.2528462, -11.7889767, 7.2531080, -17.7247238, 17.7324066
31: -22.9012012, -4.3948064, -22.9037685, -4.3892717, -15.2644043, 15.2570534
32: -3.7873406, 10.6064835, -3.7935078, 10.6010427, -11.2141762, 11.2496071
33: 10.5334654, 30.9043007, 10.5189514, 30.8852253, -16.2461624, 16.2823944
34: 11.2965899, 29.0108490, 11.2811050, 28.9862404, -11.3798218, 11.4249268
35: 22.9453583, 40.5146446, 22.9303360, 40.4910812, -11.2930565, 11.2858276
36: 17.9713573, 34.5469513, 17.9571953, 34.5309067, -12.3297539, 12.3622055
37: 7.8784528, 28.1284447, 7.8732867, 28.1271057, -16.7474289, 16.7342987
38: 6.6133151, 26.6066093, 6.5986452, 26.5954361, -14.3814621, 14.3976097
39: 5.7215433, 25.9483891, 5.7110543, 25.9471970, -16.1997299, 16.2194519
40: 0.6060028, 19.8796768, 0.5976534, 19.8650970, -12.5676727, 12.6168633
41: -4.0758495, 9.0985060, -4.0822992, 9.0901403, -10.9454651, 10.9820328
42: -27.5754185, -10.8450251, -27.5804558, -10.8490448, -11.5132217, 11.5340309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 695

## Relational analysis of IS_B2_A2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1881113, upper bound: 5.2139379
time: 6.05 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1953728, upper bound: 5.2139379
time: 5.52 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -57.6873894, -32.6120377, -57.6458168, -32.6123657, -17.5466766, 17.5055847
1: -39.2195930, -20.1958103, -39.1884499, -20.1958370, -11.8994293, 11.8649559
2: -27.2546463, -11.1536617, -27.2312450, -11.1548214, -11.0300713, 11.0037231
3: -31.5754852, -14.0772352, -31.5724297, -14.0780354, -10.9583473, 10.9545555
4: -29.4217110, -8.6269169, -29.3901920, -8.6282501, -14.1944809, 14.1755142
5: -31.7602921, -13.5342159, -31.7474861, -13.5332832, -12.1844635, 12.1663666
6: -14.8925858, 2.9226561, -14.8952141, 2.8866148, -11.5682106, 11.6368027
7: -46.6821251, -25.5293407, -46.6414261, -25.5289421, -12.0482368, 12.0006294
8: -41.4715347, -19.8279495, -41.4306259, -19.8290710, -10.7205276, 10.6833591
9: -24.3024826, -5.1247416, -24.2791557, -5.1263542, -16.4882660, 16.4564896
10: -52.1512718, -29.6357536, -52.0949860, -29.6404419, -17.1078911, 17.0985184
11: -47.9273224, -27.0895271, -47.9120255, -27.0878410, -15.0625153, 15.0974617
12: -13.3577719, 5.9207025, -13.3602009, 5.9123421, -15.4053001, 15.4421501
13: -9.2970200, 9.7560749, -9.2768888, 9.7551622, -16.3221588, 16.3240051
14: -86.2000504, -59.4821014, -86.0907135, -59.4812546, -20.0143738, 19.8481636
15: -29.5849628, -11.9110374, -29.5699387, -11.9123631, -12.1595116, 12.1393356
16: -43.4125786, -22.5500412, -43.3765106, -22.5497589, -16.2545547, 16.2265053
17: -100.0459671, -70.0140762, -99.9715805, -70.0141754, -22.1759491, 22.1424179
18: -17.7606201, 3.4611270, -17.7582512, 3.4593165, -13.6948318, 13.6884804
19: -21.0303898, -6.4484844, -21.0198059, -6.4440618, -12.4120102, 12.4355698
20: -8.1969919, 5.5892763, -8.1928911, 5.5881386, -13.7851305, 13.7821674
21: -30.5002575, -12.1469240, -30.4817085, -12.1434832, -16.0932617, 16.1266785
22: -24.8135891, -8.3604393, -24.8054333, -8.3594904, -12.1532784, 12.1758728
23: -16.8913918, 0.1569176, -16.8783989, 0.1585647, -14.1354904, 14.1562004
24: -8.0180416, 6.9065971, -8.0157948, 6.9014788, -12.7794647, 12.7981682
25: -4.5986471, 11.7324715, -4.5924664, 11.7310476, -14.1714554, 14.1832314
26: -23.0608902, -1.5587254, -23.0572262, -1.5659659, -18.3025589, 18.3299866
27: -17.8072224, -3.7810082, -17.8058243, -3.7875223, -12.8964081, 12.9077721
28: -3.3369257, 16.1773167, -3.3322899, 16.1681423, -15.9788895, 15.9825745
29: -41.7422714, -23.3513126, -41.7349243, -23.3528309, -14.5399704, 14.5578842
30: -11.8003178, 7.2683973, -11.7954597, 7.2540145, -17.7402191, 17.7627258
31: -22.9143867, -4.3972645, -22.9096584, -4.3910065, -15.2762070, 15.2644844
32: -3.8005064, 10.6104288, -3.7998586, 10.6008587, -11.2224312, 11.2597618
33: 10.4986305, 30.9267426, 10.5009117, 30.8856697, -16.2745590, 16.3223343
34: 11.2685032, 29.0374622, 11.2669010, 28.9867172, -11.4033089, 11.4651489
35: 22.9100857, 40.5407066, 22.9120674, 40.4911423, -11.3208084, 11.3292885
36: 17.9389782, 34.5645828, 17.9400558, 34.5310135, -12.3555031, 12.3962669
37: 7.8643775, 28.1348114, 7.8661947, 28.1273842, -16.7601852, 16.7467613
38: 6.5859113, 26.6154137, 6.5841780, 26.5962009, -14.4076614, 14.4197044
39: 5.6935372, 25.9494305, 5.6968870, 25.9455605, -16.2272949, 16.2371521
40: 0.5896802, 19.8958244, 0.5889130, 19.8663044, -12.5792770, 12.6394463
41: -4.0883417, 9.1119003, -4.0883484, 9.0907841, -10.9544487, 11.0007782
42: -27.5832443, -10.8330250, -27.5843067, -10.8476887, -11.5186501, 11.5502815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 695

## Relational analysis of IS_B2_A2_A2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2073274, upper bound: 5.2145847
time: 5.37 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2145845, upper bound: 5.2145847
time: 6.00 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.45 seconds
IS_B1_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2053408, upper bound: 5.1740398
IS_B1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2126050, upper bound: 5.1740398
IS_B1_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2059115, upper bound: 5.1932912
IS_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2131746, upper bound: 5.1932912
IS_B2_A1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2063938, upper bound: 5.1938078
IS_B2_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2136589, upper bound: 5.1938078
IS_B2_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1732593, upper bound: 5.2046499
IS_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1732593, upper bound: 5.2119125
IS_B2_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1899624, upper bound: 5.2067967
IS_B2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1899624, upper bound: 5.2140596
IS_B2_A2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1898637, upper bound: 5.2054040
IS_B2_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1898637, upper bound: 5.2126665
IS_B2_A2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1881113, upper bound: 5.2139379
IS_B2_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.1953728, upper bound: 5.2139379
IS_B2_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2073274, upper bound: 5.2145847
IS_B2_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 35, lower bound: -5.2145845, upper bound: 5.2145847

## BFS IS instance: IS_B1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -57.6540298, -32.6600151, -57.6112671, -32.6443176, -17.5103416, 17.4165878
1: -39.2001305, -20.2304535, -39.1699867, -20.2178879, -11.8804703, 11.8086395
2: -27.2405281, -11.1782045, -27.2169724, -11.1547089, -11.0295792, 10.9540443
3: -31.5705032, -14.0910177, -31.5614395, -14.0716515, -10.9422684, 10.9098167
4: -29.4043598, -8.6555557, -29.3724384, -8.6334686, -14.2017365, 14.0983772
5: -31.7499561, -13.5606890, -31.7366600, -13.5367756, -12.1621704, 12.1106758
6: -14.8610916, 2.8989973, -14.8305416, 2.8661137, -11.5330391, 11.5740242
7: -46.6563301, -25.5771809, -46.6185150, -25.5678539, -12.0094872, 11.9253273
8: -41.4502907, -19.8928566, -41.4118576, -19.9223080, -10.6482849, 10.5972519
9: -24.2816582, -5.1466117, -24.2556534, -5.1346674, -16.4611816, 16.4121628
10: -52.1129112, -29.7039280, -52.0543137, -29.6948280, -17.1163712, 16.9534149
11: -47.9119835, -27.1166573, -47.8949203, -27.1189270, -15.0109253, 15.0930862
12: -13.3354836, 5.9090204, -13.3284149, 5.8954525, -15.3781242, 15.3499985
13: -9.2561092, 9.7330580, -9.2219448, 9.7746716, -16.3271866, 16.2054596
14: -86.1313705, -59.5788727, -86.0345306, -59.6470871, -19.8138123, 19.7554779
15: -29.5723763, -11.9345484, -29.5602531, -11.9334621, -12.1187477, 12.0998917
16: -43.3758316, -22.5956116, -43.3475418, -22.5591011, -16.2305908, 16.1641159
17: -99.9922333, -70.0915604, -99.9312439, -70.0868301, -21.9918060, 22.1273346
18: -17.7519417, 3.4580293, -17.7942085, 3.4507296, -13.6325073, 13.7369843
19: -21.0061264, -6.4740825, -20.9884434, -6.4434261, -12.4286880, 12.3890953
20: -8.1848536, 5.5683012, -8.1986771, 5.5538907, -13.7387447, 13.7669783
21: -30.4755630, -12.1774559, -30.4531326, -12.1587076, -16.0767441, 16.0859528
22: -24.7875080, -8.3697472, -24.7762680, -8.3648109, -12.1121368, 12.1535645
23: -16.8734207, 0.1231626, -16.8594627, 0.1001648, -14.0700455, 14.1369896
24: -8.0054283, 6.8999701, -8.0376282, 6.8889728, -12.7342682, 12.8146133
25: -4.5812368, 11.7163544, -4.5788798, 11.7029591, -14.1202774, 14.1499481
26: -23.0453758, -1.5767446, -23.0940247, -1.5952966, -18.1979675, 18.3660355
27: -17.7989807, -3.7914023, -17.8411961, -3.8039150, -12.8369827, 12.9404984
28: -3.3196034, 16.1667538, -3.3329105, 16.1480217, -15.9342117, 15.9556122
29: -41.7296715, -23.3668194, -41.7235413, -23.3384056, -14.5137482, 14.5494652
30: -11.7934189, 7.2578082, -11.8527889, 7.2404094, -17.6953659, 17.7970581
31: -22.8850060, -4.4175062, -22.8662605, -4.3913622, -15.2647095, 15.2206802
32: -3.7663448, 10.6013985, -3.7412677, 10.5946350, -11.2511101, 11.1653862
33: 10.5704737, 30.9044342, 10.6157751, 30.8699093, -16.2026596, 16.2040939
34: 11.3240805, 29.0035973, 11.2756548, 28.9570885, -11.3066978, 11.4592552
35: 22.9852619, 40.5148697, 23.0025444, 40.4684143, -11.1827469, 11.2958717
36: 18.0161362, 34.5434875, 18.0194244, 34.5102959, -12.2651100, 12.3224144
37: 7.8823962, 28.1339455, 7.8536224, 28.1225586, -16.6901855, 16.7687531
38: 6.6539011, 26.5890846, 6.6539383, 26.5682812, -14.3042946, 14.3478127
39: 5.7686691, 25.9491234, 5.8197446, 25.9494724, -16.1633682, 16.0983200
40: 0.6217737, 19.8723907, 0.5951433, 19.8474674, -12.5392914, 12.5751419
41: -4.0673399, 9.0989323, -4.0497551, 9.0808907, -10.9330063, 10.9334831
42: -27.5777283, -10.8481178, -27.5712070, -10.8579350, -11.4988785, 11.5177422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=112, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B1_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_B2_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2018698, upper bound: 5.1721453
time: 11.80 seconds

## Relational analysis of IS_B1_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_B2_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2107114, upper bound: 5.1721453
time: 28.05 seconds

## BFS IS instance: IS_B1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -57.6571198, -32.6436539, -57.6378212, -32.6170540, -17.5348816, 17.4630318
1: -39.2009354, -20.2173576, -39.1869049, -20.1949825, -11.8977013, 11.8397675
2: -27.2415257, -11.1700459, -27.2283249, -11.1402035, -11.0428696, 10.9745216
3: -31.5706482, -14.0863800, -31.5642834, -14.0630112, -10.9541931, 10.9209785
4: -29.4052868, -8.6454201, -29.3849144, -8.6151295, -14.2154846, 14.1229172
5: -31.7506924, -13.5512447, -31.7451019, -13.5205517, -12.1792297, 12.1294441
6: -14.8732786, 2.8996878, -14.8522186, 2.8840122, -11.5660210, 11.5860100
7: -46.6570625, -25.5603905, -46.6420784, -25.5387878, -12.0313950, 11.9656410
8: -41.4507027, -19.8710384, -41.4320488, -19.8841438, -10.6787186, 10.6395187
9: -24.2825451, -5.1401296, -24.2677155, -5.1222363, -16.4700928, 16.4324951
10: -52.1131287, -29.6823311, -52.0827065, -29.6551895, -17.1380997, 17.0064850
11: -47.9128685, -27.1101170, -47.9079742, -27.1070843, -15.0227890, 15.0877609
12: -13.3400135, 5.9108090, -13.3371220, 5.9034352, -15.3982773, 15.3672752
13: -9.2658062, 9.7419300, -9.2468243, 9.7909813, -16.3526611, 16.2404976
14: -86.1340332, -59.5446663, -86.0989227, -59.5884171, -19.8502197, 19.8551826
15: -29.5746307, -11.9264107, -29.5685863, -11.9187260, -12.1350288, 12.1175728
16: -43.3778572, -22.5808277, -43.3728256, -22.5338669, -16.2522659, 16.1973152
17: -99.9934921, -70.0616150, -99.9778061, -70.0342636, -22.0246429, 22.1694565
18: -17.7547607, 3.4561207, -17.7991009, 3.4506381, -13.6427994, 13.7396545
19: -21.0094337, -6.4682946, -21.0023766, -6.4326968, -12.4371758, 12.3935966
20: -8.1885748, 5.5721169, -8.2074909, 5.5612721, -13.7498474, 13.7796078
21: -30.4778061, -12.1700296, -30.4710770, -12.1456594, -16.0880585, 16.0888214
22: -24.7907829, -8.3668098, -24.7877483, -8.3583879, -12.1234436, 12.1607780
23: -16.8757305, 0.1316762, -16.8723278, 0.1163755, -14.0827713, 14.1460991
24: -8.0094652, 6.9009500, -8.0454636, 6.8924513, -12.7477417, 12.8222809
25: -4.5855155, 11.7194805, -4.5905914, 11.7097530, -14.1299744, 14.1596909
26: -23.0501518, -1.5752506, -23.1041317, -1.5876074, -18.2266083, 18.3754654
27: -17.8020859, -3.7902074, -17.8472862, -3.8000073, -12.8486710, 12.9460983
28: -3.3247781, 16.1669254, -3.3442135, 16.1506500, -15.9440002, 15.9688110
29: -41.7314606, -23.3619232, -41.7315331, -23.3285637, -14.5228653, 14.5528107
30: -11.7952175, 7.2597995, -11.8580875, 7.2453079, -17.7087479, 17.8067169
31: -22.8915043, -4.4133806, -22.8786736, -4.3837376, -15.2766113, 15.2312813
32: -3.7768788, 10.6019354, -3.7598326, 10.6019211, -11.2664909, 11.1809883
33: 10.5457134, 30.9051132, 10.5719519, 30.8912067, -16.2493668, 16.2381210
34: 11.3031921, 29.0042610, 11.2391329, 28.9836788, -11.3541603, 11.4876823
35: 22.9601383, 40.5148315, 22.9584141, 40.4929352, -11.2322044, 11.3287544
36: 17.9904594, 34.5435600, 17.9742813, 34.5271378, -12.3063660, 12.3577118
37: 7.8765106, 28.1319599, 7.8419251, 28.1215096, -16.7018127, 16.7807388
38: 6.6289549, 26.5899696, 6.6096182, 26.5819607, -14.3410034, 14.3891029
39: 5.7468395, 25.9486446, 5.7803712, 25.9520206, -16.1919250, 16.1394043
40: 0.6082144, 19.8739338, 0.5712833, 19.8661366, -12.5723267, 12.5915947
41: -4.0746326, 9.0998707, -4.0626636, 9.0916862, -10.9514656, 10.9451485
42: -27.5808525, -10.8452787, -27.5767441, -10.8481331, -11.5116348, 11.5239525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=112, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B1_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_B1_B2_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2024406, upper bound: 5.1913972
time: 6.09 seconds

## Relational analysis of IS_B1_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_B1_B2_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2112808, upper bound: 5.1913972
time: 16.33 seconds

## BFS IS instance: IS_B2_A1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -57.6676025, -32.6203880, -57.6323624, -32.5646057, -17.5750732, 17.4787560
1: -39.2080078, -20.2027283, -39.1808395, -20.1594887, -11.9208565, 11.8378105
2: -27.2457695, -11.1578827, -27.2247868, -11.1140079, -11.0662842, 10.9833336
3: -31.5716915, -14.0833950, -31.5697346, -14.0514421, -10.9736404, 10.9382095
4: -29.3908272, -8.6531467, -29.3714485, -8.5864429, -14.2177391, 14.1219978
5: -31.7565765, -13.5412960, -31.7451458, -13.4927616, -12.2140045, 12.1399384
6: -14.8854923, 2.9173207, -14.8908482, 2.8825483, -11.5603600, 11.6241875
7: -46.6711655, -25.5350342, -46.6343498, -25.4879494, -12.0804863, 11.9805222
8: -41.4620094, -19.8353844, -41.4254227, -19.8133259, -10.7339630, 10.6705017
9: -24.2672215, -5.1499887, -24.2521114, -5.1007910, -16.4687653, 16.4025497
10: -52.0953445, -29.6822929, -52.0516319, -29.5842323, -17.1550064, 17.0080719
11: -47.9180145, -27.1027622, -47.9105759, -27.0781574, -15.0056152, 15.0892639
12: -13.3142185, 5.8933783, -13.3343277, 5.9094071, -15.3918076, 15.3892479
13: -9.2886562, 9.7434196, -9.2750769, 9.8136044, -16.3879089, 16.2728195
14: -86.1320648, -59.5325127, -86.0547638, -59.4860306, -19.9331360, 19.8337517
15: -29.5634575, -11.9286213, -29.5641518, -11.8957424, -12.1443481, 12.1154022
16: -43.3867722, -22.5613632, -43.3588676, -22.4852943, -16.2948532, 16.2007256
17: -100.0064545, -70.0305634, -99.9514465, -69.9541931, -22.0875397, 22.1561890
18: -17.7325211, 3.4424727, -17.7934170, 3.4564176, -13.6451683, 13.7335510
19: -21.0112381, -6.4647226, -21.0119400, -6.4046659, -12.4320297, 12.3931313
20: -8.1861839, 5.5817900, -8.2156525, 5.5818100, -13.7679939, 13.7974424
21: -30.4719696, -12.1768560, -30.4766464, -12.1199884, -16.0773239, 16.0796013
22: -24.7972431, -8.3617554, -24.7942944, -8.3477774, -12.1283684, 12.1716576
23: -16.8615074, 0.1196703, -16.8767815, 0.1343231, -14.0739441, 14.1361351
24: -8.0106411, 6.8998175, -8.0548573, 6.8983822, -12.7369385, 12.8249779
25: -4.5687666, 11.7006321, -4.6024714, 11.7131510, -14.1119843, 14.1488457
26: -23.0329742, -1.5723381, -23.1068306, -1.5752609, -18.2404022, 18.3942032
27: -17.8003578, -3.7857890, -17.8582573, -3.7922981, -12.8619843, 12.9617882
28: -3.3167419, 16.1523972, -3.3580108, 16.1489258, -15.9351120, 15.9681702
29: -41.7293205, -23.3630619, -41.7346764, -23.3185825, -14.5281906, 14.5608025
30: -11.7781992, 7.2400336, -11.8611050, 7.2386189, -17.6871948, 17.7981949
31: -22.8964310, -4.3977675, -22.8930359, -4.3468661, -15.2810593, 15.2310104
32: -3.7588518, 10.5798426, -3.7723031, 10.6008692, -11.2439651, 11.1868324
33: 10.5278168, 30.8982201, 10.4988546, 30.8711739, -16.2718353, 16.3012466
34: 11.2737112, 29.0300102, 11.1732578, 28.9804306, -11.3696327, 11.5620689
35: 22.9496250, 40.5040894, 22.8788433, 40.4681435, -11.2590714, 11.3752518
36: 17.9448776, 34.5558052, 17.8896751, 34.5237541, -12.3361664, 12.4333801
37: 7.9095235, 28.0796604, 7.8309102, 28.0937977, -16.7138901, 16.7711678
38: 6.5946898, 26.6005421, 6.5326357, 26.5801678, -14.3754845, 14.4674950
39: 5.7113709, 25.9495316, 5.7164707, 25.9545383, -16.2288742, 16.2043533
40: 0.6173687, 19.8736420, 0.5536289, 19.8633995, -12.5558472, 12.6312141
41: -4.0855956, 9.1093044, -4.0866599, 9.0922308, -10.9435387, 10.9590569
42: -27.5802174, -10.8370552, -27.5840874, -10.8403873, -11.5104065, 11.5349426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B2_A1_A2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2029229, upper bound: 5.1919209
time: 5.64 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2117654, upper bound: 5.1919209
time: 5.28 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -57.6121216, -32.6278191, -57.6551666, -32.6492119, -17.4167252, 17.5289268
1: -39.1693344, -20.2138901, -39.2007828, -20.2278538, -11.8069305, 11.8850479
2: -27.2148056, -11.1513824, -27.2415161, -11.1757965, -10.9555855, 11.0350227
3: -31.5661430, -14.0720539, -31.5729923, -14.0913029, -10.9181709, 10.9467201
4: -29.3691025, -8.6331463, -29.4060059, -8.6551533, -14.0991554, 14.2039986
5: -31.7375069, -13.5279579, -31.7508297, -13.5554571, -12.1200409, 12.1720924
6: -14.8411951, 2.8671312, -14.8673801, 2.8998051, -11.5884933, 11.5364037
7: -46.6229858, -25.5471325, -46.6573334, -25.5651436, -11.9333572, 12.0310402
8: -41.4152222, -19.8911514, -41.4507065, -19.8747330, -10.6082954, 10.6801586
9: -24.2544918, -5.1302652, -24.2828960, -5.1444740, -16.4163895, 16.4658432
10: -52.0540543, -29.6628151, -52.1131134, -29.6852722, -16.9486847, 17.1481094
11: -47.9061737, -27.0874443, -47.9120178, -27.0972176, -15.0875931, 15.0216026
12: -13.3417692, 5.8970928, -13.3405304, 5.9129515, -15.3636932, 15.3842010
13: -9.2359638, 9.7694225, -9.2671967, 9.7300358, -16.2210999, 16.3334122
14: -86.0430984, -59.6027908, -86.1340179, -59.5530319, -19.7713318, 19.8599014
15: -29.5571690, -11.9364290, -29.5745544, -11.9358215, -12.1028557, 12.1185913
16: -43.3485870, -22.5379314, -43.3769836, -22.5815239, -16.1732483, 16.2566948
17: -99.9386063, -70.0565796, -99.9948502, -70.0742188, -22.1306381, 22.0442200
18: -17.7955856, 3.4551859, -17.7542591, 3.4593580, -13.7386246, 13.6376648
19: -20.9994659, -6.4093571, -21.0069561, -6.4517479, -12.3843079, 12.4524078
20: -8.1940269, 5.5725689, -8.1826391, 5.5798674, -13.7738943, 13.7552080
21: -30.4677238, -12.1183033, -30.4761028, -12.1521425, -16.0757446, 16.0986404
22: -24.7840614, -8.3542662, -24.7895794, -8.3625431, -12.1507874, 12.1206932
23: -16.8656712, 0.1361059, -16.8734646, 0.1450899, -14.1287689, 14.0949516
24: -8.0298529, 6.8921609, -8.0014038, 6.9029722, -12.8085938, 12.7286949
25: -4.5807657, 11.7207108, -4.5809441, 11.7268314, -14.1472626, 14.1298981
26: -23.0918846, -1.5859084, -23.0446320, -1.5743566, -18.3702621, 18.2033691
27: -17.8372154, -3.8000646, -17.7972221, -3.7885287, -12.9403076, 12.8364449
28: -3.3255723, 16.1484966, -3.3154607, 16.1688118, -15.9517975, 15.9332504
29: -41.7271767, -23.3235741, -41.7301178, -23.3567505, -14.5454865, 14.5209427
30: -11.8390265, 7.2363806, -11.7848701, 7.2612696, -17.7890472, 17.6838684
31: -22.8669910, -4.3578758, -22.8870697, -4.3963666, -15.2218018, 15.2925987
32: -3.7560389, 10.5981417, -3.7750473, 10.6024666, -11.1766853, 11.2518425
33: 10.5892715, 30.8675098, 10.5544291, 30.9053173, -16.2304764, 16.2099762
34: 11.2509432, 28.9607067, 11.3095713, 29.0049038, -11.4848022, 11.3202209
35: 22.9677887, 40.4691391, 22.9649162, 40.5151672, -11.3301468, 11.1927032
36: 17.9778347, 34.5129547, 17.9921799, 34.5438080, -12.3622055, 12.2821350
37: 7.8607612, 28.1226120, 7.8857975, 28.1336403, -16.7686157, 16.7019081
38: 6.6173601, 26.5719872, 6.6331301, 26.5897808, -14.3882065, 14.3251266
39: 5.7920003, 25.9497375, 5.7521458, 25.9471970, -16.1311798, 16.1855164
40: 0.5912800, 19.8485203, 0.6193838, 19.8739166, -12.5851173, 12.5433617
41: -4.0556293, 9.0834627, -4.0709219, 9.1002111, -10.9440346, 10.9429016
42: -27.5636597, -10.8535385, -27.5735283, -10.8447590, -11.5136452, 11.5078773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A1_A1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1713664, upper bound: 5.2011801
time: 35.27 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1713664, upper bound: 5.2100186
time: 5.61 seconds

## BFS IS instance: IS_B2_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -57.6352997, -32.5930519, -57.6596146, -32.6287041, -17.4603806, 17.5481071
1: -39.1844330, -20.1824226, -39.2020035, -20.2092400, -11.8396568, 11.8988495
2: -27.2267342, -11.1291752, -27.2425270, -11.1628609, -10.9804268, 11.0466080
3: -31.5681515, -14.0591946, -31.5731068, -14.0836153, -10.9299164, 10.9612198
4: -29.3853569, -8.6039600, -29.4067039, -8.6380272, -14.1319656, 14.2122765
5: -31.7440414, -13.5094528, -31.7513466, -13.5439310, -12.1374207, 12.1879807
6: -14.8685045, 2.8817091, -14.8833733, 2.9005871, -11.5826874, 11.5670242
7: -46.6378670, -25.5175686, -46.6578712, -25.5473270, -11.9660187, 12.0408745
8: -41.4294815, -19.8513947, -41.4512405, -19.8510284, -10.6451225, 10.6953049
9: -24.2657166, -5.1118484, -24.2834549, -5.1334877, -16.4390030, 16.4718018
10: -52.0771866, -29.6205406, -52.1134262, -29.6605320, -16.9979591, 17.1581268
11: -47.9107933, -27.0860748, -47.9138451, -27.0962105, -15.0889587, 15.0495987
12: -13.3444443, 5.9048252, -13.3446045, 5.9151940, -15.3803978, 15.4059639
13: -9.2585297, 9.7994547, -9.2752066, 9.7472286, -16.2611389, 16.3668213
14: -86.0848999, -59.5453148, -86.1367798, -59.5187187, -19.8452835, 19.8589745
15: -29.5704632, -11.9099684, -29.5765877, -11.9203815, -12.1313477, 12.1348038
16: -43.3648300, -22.5126095, -43.3796501, -22.5661049, -16.1998520, 16.2731628
17: -99.9658890, -70.0069504, -99.9960938, -70.0445633, -22.1213531, 22.0892639
18: -17.8017159, 3.4574678, -17.7573166, 3.4601710, -13.7434540, 13.6570473
19: -21.0072746, -6.4105234, -21.0121765, -6.4518676, -12.3890724, 12.4732361
20: -8.2110405, 5.5758905, -8.1915169, 5.5811224, -13.7921629, 13.7674074
21: -30.4751091, -12.1198387, -30.4798889, -12.1526833, -16.0801773, 16.1325378
22: -24.7896805, -8.3562288, -24.7938061, -8.3636398, -12.1542244, 12.1513786
23: -16.8739929, 0.1372651, -16.8776474, 0.1456733, -14.1341629, 14.1234398
24: -8.0494194, 6.8972716, -8.0127773, 6.9035425, -12.8261795, 12.7604408
25: -4.5960941, 11.7230005, -4.5893030, 11.7271385, -14.1607742, 14.1552544
26: -23.1091099, -1.5777981, -23.0538216, -1.5708447, -18.3854065, 18.2546234
27: -17.8521767, -3.7946334, -17.8051128, -3.7871401, -12.9539032, 12.8638458
28: -3.3495791, 16.1584320, -3.3291252, 16.1706905, -15.9765244, 15.9562149
29: -41.7323532, -23.3226662, -41.7328377, -23.3564110, -14.5514374, 14.5433769
30: -11.8594999, 7.2483873, -11.7960796, 7.2626061, -17.8119278, 17.7171097
31: -22.8833103, -4.3604002, -22.8963547, -4.3975067, -15.2371750, 15.3109741
32: -3.7736115, 10.6013937, -3.7851677, 10.6029568, -11.1933479, 11.2654190
33: 10.5401936, 30.8854485, 10.5256481, 30.9059753, -16.2558823, 16.2542038
34: 11.2124290, 28.9804192, 11.2864647, 29.0056229, -11.5047951, 11.3611717
35: 22.9215736, 40.4880981, 22.9370193, 40.5152206, -11.3505592, 11.2388573
36: 17.9351349, 34.5256424, 17.9663391, 34.5440140, -12.3835793, 12.3194542
37: 7.8399329, 28.1283836, 7.8732738, 28.1339874, -16.7809677, 16.7186432
38: 6.5786171, 26.5797005, 6.6093950, 26.5911026, -14.4247627, 14.3526611
39: 5.7534151, 25.9527283, 5.7295542, 25.9478645, -16.1696930, 16.2109909
40: 0.5645933, 19.8612022, 0.6035128, 19.8756599, -12.6030617, 12.5600433
41: -4.0716887, 9.0924072, -4.0801225, 9.1011257, -10.9470787, 10.9600983
42: -27.5765133, -10.8438015, -27.5808144, -10.8428040, -11.5161781, 11.5248680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A1_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1880682, upper bound: 5.2033268
time: 23.88 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1880682, upper bound: 5.2121658
time: 26.82 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -57.6401367, -32.5987549, -57.6582870, -32.6325912, -17.4635849, 17.5540314
1: -39.1865540, -20.1904678, -39.2016182, -20.2145348, -11.8385658, 11.9027519
2: -27.2263947, -11.1358204, -27.2425823, -11.1672850, -10.9765015, 11.0485916
3: -31.5696335, -14.0629940, -31.5733852, -14.0865545, -10.9297142, 10.9591255
4: -29.3825932, -8.6143827, -29.4069099, -8.6448431, -14.1240807, 14.2179947
5: -31.7459564, -13.5113268, -31.7515736, -13.5458660, -12.1390343, 12.1895218
6: -14.8632507, 2.8851514, -14.8797092, 2.9004688, -11.6008263, 11.5696487
7: -46.6466827, -25.5175018, -46.6580734, -25.5482178, -11.9740295, 12.0534859
8: -41.4355850, -19.8522663, -41.4510841, -19.8526592, -10.6509628, 10.7112713
9: -24.2668114, -5.1169729, -24.2837906, -5.1374111, -16.4361343, 16.4752426
10: -52.0826149, -29.6227951, -52.1133003, -29.6635799, -17.0020676, 17.1701660
11: -47.9195824, -27.0751953, -47.9129486, -27.0906677, -15.0820618, 15.0337715
12: -13.3508987, 5.9063888, -13.3451519, 5.9150243, -15.3817940, 15.4051323
13: -9.2614098, 9.7866211, -9.2769661, 9.7394886, -16.2569885, 16.3597870
14: -86.1078796, -59.5436668, -86.1367188, -59.5186615, -19.8716583, 19.8968468
15: -29.5656357, -11.9205379, -29.5768375, -11.9271021, -12.1209145, 12.1353912
16: -43.3764801, -22.5101223, -43.3790436, -22.5658455, -16.2045822, 16.2787476
17: -99.9850845, -70.0037231, -99.9961700, -70.0440369, -22.1764221, 22.0748825
18: -17.8008270, 3.4552963, -17.7570992, 3.4574254, -13.7416382, 13.6482697
19: -21.0140533, -6.3977885, -21.0103264, -6.4455523, -12.3897133, 12.4617424
20: -8.2032375, 5.5807581, -8.1864452, 5.5838356, -13.7870731, 13.7672033
21: -30.4857330, -12.1047602, -30.4783440, -12.1448059, -16.0790405, 16.1120682
22: -24.7956352, -8.3469324, -24.7928848, -8.3588257, -12.1585732, 12.1320229
23: -16.8807068, 0.1526753, -16.8758469, 0.1537477, -14.1391373, 14.1078873
24: -8.0378819, 6.8960342, -8.0055256, 6.9039288, -12.8164902, 12.7421951
25: -4.5933075, 11.7290516, -4.5852714, 11.7311363, -14.1577606, 14.1403503
26: -23.1023407, -1.5779748, -23.0494804, -1.5728414, -18.3801956, 18.2328339
27: -17.8434925, -3.7959986, -17.8003712, -3.7873549, -12.9461136, 12.8483734
28: -3.3372149, 16.1515408, -3.3208106, 16.1690369, -15.9658356, 15.9439774
29: -41.7358437, -23.3133907, -41.7319221, -23.3518410, -14.5492935, 14.5307159
30: -11.8445683, 7.2415581, -11.7867622, 7.2632017, -17.7990417, 17.6976929
31: -22.8811111, -4.3500271, -22.8937492, -4.3922243, -15.2330627, 15.3050880
32: -3.7750604, 10.6056271, -3.7857518, 10.6030273, -11.1926575, 11.2675362
33: 10.5448246, 30.8889446, 10.5294304, 30.9060249, -16.2651749, 16.2570953
34: 11.2139606, 28.9873466, 11.2885170, 29.0055656, -11.5136719, 11.3679237
35: 22.9231052, 40.4936523, 22.9396591, 40.5151405, -11.3636017, 11.2423515
36: 17.9319878, 34.5298080, 17.9662609, 34.5438881, -12.3981895, 12.3236732
37: 7.8486266, 28.1217232, 7.8797922, 28.1316700, -16.7810669, 16.7138329
38: 6.5723705, 26.5857716, 6.6080027, 26.5907135, -14.4302330, 14.3623734
39: 5.7518406, 25.9522724, 5.7300878, 25.9466934, -16.1730881, 16.2145309
40: 0.5669737, 19.8672504, 0.6056442, 19.8755169, -12.6020508, 12.5780945
41: -4.0687265, 9.0943403, -4.0783138, 9.1011591, -10.9558945, 10.9615402
42: -27.5695820, -10.8435516, -27.5768986, -10.8420277, -11.5199661, 11.5209732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1879697, upper bound: 5.2019343
time: 5.85 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1879697, upper bound: 5.2107725
time: 15.53 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -57.6489258, -32.6369057, -57.6176376, -32.6276207, -17.4933395, 17.4609604
1: -39.1931915, -20.2174015, -39.1706352, -20.2094116, -11.8613968, 11.8369026
2: -27.2267704, -11.1710148, -27.2063828, -11.1655626, -10.9927750, 10.9682007
3: -31.5631752, -14.0875034, -31.5535946, -14.0853634, -10.9333267, 10.9232674
4: -29.3867683, -8.6483727, -29.3641319, -8.6414337, -14.1491241, 14.1386185
5: -31.7410011, -13.5478878, -31.7229385, -13.5426664, -12.1531906, 12.1263428
6: -14.8708649, 2.8988695, -14.8831701, 2.8800244, -11.5478058, 11.5968590
7: -46.6490021, -25.5511589, -46.6148758, -25.5433617, -12.0000801, 11.9582901
8: -41.4445190, -19.8606205, -41.4211197, -19.8483009, -10.6753845, 10.6514454
9: -24.2788715, -5.1387186, -24.2611198, -5.1354561, -16.4579391, 16.4300766
10: -52.1022644, -29.6711617, -52.0575714, -29.6625462, -17.0402527, 17.0459061
11: -47.9131775, -27.0932922, -47.8951988, -27.0925255, -15.0303497, 15.0514565
12: -13.3524456, 5.9084330, -13.3522577, 5.9046330, -15.3824234, 15.4154739
13: -9.2553310, 9.7328672, -9.2349434, 9.7410049, -16.2685547, 16.2650223
14: -86.1394653, -59.5238419, -86.0781631, -59.5063057, -19.9165039, 19.7974815
15: -29.5648117, -11.9295216, -29.5567932, -11.9237852, -12.1311302, 12.1165771
16: -43.3784180, -22.5682545, -43.3400574, -22.5621262, -16.2078323, 16.1785774
17: -99.9820709, -70.0439835, -99.9126968, -70.0311584, -22.1093674, 22.0661163
18: -17.7495651, 3.4397686, -17.7494297, 3.4244199, -13.6590958, 13.6565247
19: -21.0079460, -6.4469357, -20.9893951, -6.4430580, -12.3886375, 12.3928146
20: -8.1839123, 5.5801296, -8.1838417, 5.5746756, -13.7585878, 13.7639713
21: -30.4767990, -12.1460342, -30.4519424, -12.1437807, -16.0688934, 16.0748100
22: -24.7968464, -8.3607588, -24.7854767, -8.3598022, -12.1349564, 12.1422882
23: -16.8822708, 0.1564887, -16.8717690, 0.1574596, -14.1243973, 14.1330070
24: -8.0017786, 6.8891773, -8.0054197, 6.8789463, -12.7471008, 12.7608070
25: -4.5849628, 11.7272463, -4.5822034, 11.7240219, -14.1546555, 14.1613808
26: -23.0435295, -1.5775957, -23.0432415, -1.5912280, -18.2634277, 18.2732544
27: -17.7916069, -3.8047471, -17.7959442, -3.8191972, -12.8546448, 12.8655052
28: -3.3163588, 16.1596546, -3.3188620, 16.1535034, -15.9494858, 15.9539032
29: -41.7216339, -23.3546753, -41.7053680, -23.3540897, -14.5133209, 14.5233612
30: -11.7825584, 7.2320042, -11.7834949, 7.2148438, -17.6856079, 17.7094193
31: -22.8876495, -4.3955526, -22.8782024, -4.3906527, -15.2438431, 15.2205276
32: -3.7855539, 10.6020317, -3.7901773, 10.5929985, -11.1939049, 11.2248611
33: 10.5365639, 30.9035282, 10.5246010, 30.8837357, -16.2329712, 16.2673492
34: 11.2988091, 28.9822941, 11.2851124, 28.9320927, -11.3204384, 11.3894424
35: 22.9483166, 40.5044479, 22.9357319, 40.4723358, -11.2729301, 11.2720795
36: 17.9732552, 34.5335846, 17.9607582, 34.5062218, -12.3042221, 12.3460312
37: 7.8811550, 28.1179104, 7.8783221, 28.1071358, -16.7276230, 16.7196617
38: 6.6143231, 26.5867920, 6.6005583, 26.5580101, -14.3448410, 14.3759422
39: 5.7272801, 25.9470139, 5.7215300, 25.9446564, -16.1907043, 16.2053986
40: 0.6070070, 19.8622265, 0.5994911, 19.8319492, -12.5277176, 12.5902939
41: -4.0734792, 9.0947647, -4.0780478, 9.0833569, -10.9355507, 10.9739227
42: -27.5704327, -10.8490314, -27.5710335, -10.8564072, -11.4999352, 11.5212059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1789051, upper bound: 5.2120410
time: 24.64 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1858646, upper bound: 5.2120410
time: 16.04 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -57.6595650, -32.6356049, -57.6383362, -32.5760307, -17.5511398, 17.4709816
1: -39.2006989, -20.2163200, -39.1854858, -20.1695213, -11.9029541, 11.8408852
2: -27.2380066, -11.1702251, -27.2279930, -11.1218243, -11.0471649, 10.9776917
3: -31.5724068, -14.0872841, -31.5713882, -14.0548134, -10.9660034, 10.9294243
4: -29.3977127, -8.6472740, -29.3870411, -8.5958099, -14.1889496, 14.1479912
5: -31.7524509, -13.5474176, -31.7451515, -13.4984818, -12.2042542, 12.1322327
6: -14.8709211, 2.8973980, -14.8832788, 2.8824835, -11.5538712, 11.5989380
7: -46.6615295, -25.5506840, -46.6389847, -25.4985657, -12.0569916, 11.9676819
8: -41.4487915, -19.8600025, -41.4301147, -19.8291721, -10.6964912, 10.6564198
9: -24.2806797, -5.1380501, -24.2677155, -5.1059580, -16.4771042, 16.4333038
10: -52.1114731, -29.6689720, -52.0782776, -29.5982094, -17.1060829, 17.0560303
11: -47.9207153, -27.0930347, -47.9114304, -27.0726109, -15.0432510, 15.0839844
12: -13.3468151, 5.9120531, -13.3510275, 5.9109511, -15.3811722, 15.4174118
13: -9.2707787, 9.7343712, -9.2704678, 9.8074389, -16.3471603, 16.2840576
14: -86.1429367, -59.5252571, -86.0896988, -59.5072441, -19.9200516, 19.8217316
15: -29.5692673, -11.9283867, -29.5737419, -11.9032278, -12.1358795, 12.1275826
16: -43.3899155, -22.5674763, -43.3673172, -22.4926815, -16.2881927, 16.2056122
17: -100.0076370, -70.0439301, -99.9688187, -69.9693985, -22.1513290, 22.1286240
18: -17.7518730, 3.4556804, -17.8053226, 3.4562252, -13.6790314, 13.7164536
19: -21.0169029, -6.4469166, -21.0101280, -6.3941612, -12.4612350, 12.4037628
20: -8.1862898, 5.5832558, -8.2149210, 5.5832539, -13.7695436, 13.7981768
21: -30.4875183, -12.1459036, -30.4770279, -12.1025419, -16.1251831, 16.0842896
22: -24.7952652, -8.3607197, -24.7932758, -8.3474092, -12.1486053, 12.1570854
23: -16.8841553, 0.1542140, -16.8758068, 0.1555227, -14.1250916, 14.1394844
24: -8.0031052, 6.8993378, -8.0501518, 6.9002800, -12.7625046, 12.8107834
25: -4.5870414, 11.7301073, -4.5994806, 11.7308884, -14.1523819, 14.1616516
26: -23.0467873, -1.5711026, -23.1136284, -1.5756476, -18.2820511, 18.3565063
27: -17.7934227, -3.7922339, -17.8544197, -3.7924280, -12.8731461, 12.9288483
28: -3.3186064, 16.1614380, -3.3511248, 16.1590919, -15.9501038, 15.9701385
29: -41.7337761, -23.3547592, -41.7340393, -23.3127747, -14.5349579, 14.5524750
30: -11.7851868, 7.2496986, -11.8565884, 7.2518005, -17.7140732, 17.7930450
31: -22.8922062, -4.3953304, -22.8892097, -4.3456550, -15.3246613, 15.2283592
32: -3.7821136, 10.6055918, -3.7868905, 10.6031065, -11.2424316, 11.2299423
33: 10.5349436, 30.9038467, 10.5142822, 30.8867645, -16.2508011, 16.2729645
34: 11.2971449, 29.0050964, 11.1860418, 28.9811935, -11.3512077, 11.5277596
35: 22.9467392, 40.5129623, 22.8954372, 40.4887085, -11.2828217, 11.3184624
36: 17.9726925, 34.5427933, 17.9065285, 34.5260849, -12.3163528, 12.4045143
37: 7.8804760, 28.1273994, 7.8356390, 28.1254539, -16.7358627, 16.7636566
38: 6.6142569, 26.5952110, 6.5454264, 26.5807190, -14.3593369, 14.4508514
39: 5.7340512, 25.9481544, 5.7287922, 25.9550819, -16.2035904, 16.2028809
40: 0.6070709, 19.8769379, 0.5480366, 19.8628082, -12.5406799, 12.6447334
41: -4.0736923, 9.0977888, -4.0800653, 9.0935574, -10.9376678, 10.9723358
42: -27.5745926, -10.8466263, -27.5802841, -10.8406267, -11.5137672, 11.5250320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1846425, upper bound: 5.2120410
time: 5.74 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1934783, upper bound: 5.2120410
time: 12.86 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -57.6741333, -32.6136360, -57.6206512, -32.6153030, -17.5313911, 17.4813004
1: -39.2106552, -20.1973190, -39.1714783, -20.1986160, -11.8894882, 11.8530045
2: -27.2419033, -11.1550312, -27.2070999, -11.1574364, -11.0160980, 10.9815407
3: -31.5652046, -14.0787773, -31.5529060, -14.0809517, -10.9429703, 10.9335289
4: -29.4080105, -8.6284904, -29.3646202, -8.6311226, -14.1808090, 14.1534576
5: -31.7475929, -13.5357275, -31.7233009, -13.5360594, -12.1663437, 12.1385727
6: -14.8920326, 2.9193888, -14.8941040, 2.8806152, -11.5575409, 11.6296539
7: -46.6683121, -25.5311623, -46.6152344, -25.5323982, -12.0302696, 11.9734650
8: -41.4666939, -19.8294430, -41.4214668, -19.8318005, -10.7138977, 10.6744633
9: -24.2926807, -5.1260948, -24.2613907, -5.1289263, -16.4771118, 16.4392090
10: -52.1316757, -29.6391258, -52.0577621, -29.6466408, -17.0867844, 17.0688629
11: -47.9189186, -27.0917015, -47.8963203, -27.0919228, -15.0309982, 15.0671082
12: -13.3549023, 5.9167686, -13.3547535, 5.9052429, -15.3937607, 15.4310875
13: -9.2776060, 9.7540007, -9.2404127, 9.7513790, -16.3016357, 16.2908859
14: -86.1941986, -59.4837570, -86.0800018, -59.4844437, -19.9917526, 19.8224678
15: -29.5783482, -11.9124594, -29.5580482, -11.9149961, -12.1535530, 12.1317711
16: -43.3939209, -22.5519981, -43.3417511, -22.5534782, -16.2288132, 16.1923332
17: -100.0152588, -70.0147400, -99.9135284, -70.0153198, -22.1196976, 22.0811996
18: -17.7571125, 3.4429507, -17.7518177, 3.4251313, -13.6643753, 13.6696739
19: -21.0155945, -6.4486771, -20.9925499, -6.4443555, -12.3928986, 12.4007378
20: -8.1942530, 5.5822883, -8.1878262, 5.5750751, -13.7693281, 13.7701149
21: -30.4854527, -12.1476603, -30.4541931, -12.1448822, -16.0718384, 16.0906830
22: -24.8044453, -8.3607159, -24.7881298, -8.3599701, -12.1403351, 12.1569405
23: -16.8891258, 0.1558480, -16.8742390, 0.1565568, -14.1269989, 14.1453323
24: -8.0160675, 6.8946624, -8.0122356, 6.8792391, -12.7583694, 12.7836609
25: -4.5955009, 11.7287798, -4.5866213, 11.7241316, -14.1640778, 14.1742783
26: -23.0561714, -1.5714447, -23.0485497, -1.5899379, -18.2706070, 18.3069000
27: -17.8048630, -3.7974753, -17.8015366, -3.8182876, -12.8647919, 12.8888474
28: -3.3341143, 16.1702385, -3.3270605, 16.1547947, -15.9681091, 15.9725494
29: -41.7276764, -23.3516235, -41.7072449, -23.3534431, -14.5219345, 14.5364304
30: -11.7972851, 7.2475643, -11.7900047, 7.2156992, -17.7011032, 17.7397079
31: -22.9008389, -4.3979797, -22.8840675, -4.3923960, -15.2556610, 15.2279587
32: -3.7987146, 10.6059618, -3.7965541, 10.5927811, -11.2022247, 11.2349777
33: 10.5017166, 30.9259815, 10.5065727, 30.8841839, -16.2613602, 16.3072968
34: 11.2707043, 29.0088940, 11.2708807, 28.9325352, -11.3439369, 11.4297142
35: 22.9130249, 40.5305176, 22.9174347, 40.4723854, -11.3006935, 11.3155518
36: 17.9408512, 34.5511780, 17.9436417, 34.5063286, -12.3299637, 12.3800888
37: 7.8670754, 28.1243000, 7.8712697, 28.1074085, -16.7404175, 16.7321167
38: 6.5869207, 26.5955696, 6.5860844, 26.5587387, -14.3710442, 14.3980408
39: 5.6992855, 25.9480095, 5.7072959, 25.9430332, -16.2182770, 16.2231140
40: 0.5906610, 19.8783493, 0.5907259, 19.8331528, -12.5393410, 12.6128922
41: -4.0859661, 9.1081362, -4.0841026, 9.0839834, -10.9444923, 10.9926567
42: -27.5782890, -10.8370247, -27.5748730, -10.8550634, -11.5053444, 11.5374565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1981265, upper bound: 5.2126842
time: 5.29 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2050768, upper bound: 5.2126842
time: 21.21 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -57.6847343, -32.6123314, -57.6413536, -32.5637207, -17.5891724, 17.4912910
1: -39.2181740, -20.1962242, -39.1863174, -20.1586952, -11.9310455, 11.8570023
2: -27.2531834, -11.1542406, -27.2286797, -11.1136942, -11.0704765, 10.9910545
3: -31.5744095, -14.0785341, -31.5707703, -14.0504112, -10.9756508, 10.9396896
4: -29.4189682, -8.6274204, -29.3875046, -8.5855169, -14.2205849, 14.1628571
5: -31.7590637, -13.5352774, -31.7455330, -13.4918642, -12.2174072, 12.1444778
6: -14.8920803, 2.9179564, -14.8942175, 2.8830309, -11.5636139, 11.6317444
7: -46.6808281, -25.5306511, -46.6393471, -25.4876060, -12.0871887, 11.9828682
8: -41.4709816, -19.8288193, -41.4304504, -19.8126640, -10.7350235, 10.6794415
9: -24.2944565, -5.1254115, -24.2679672, -5.0994439, -16.4962921, 16.4424744
10: -52.1408997, -29.6369724, -52.0784607, -29.5823231, -17.1525345, 17.0789871
11: -47.9264374, -27.0914326, -47.9125595, -27.0720329, -15.0439072, 15.0996475
12: -13.3492794, 5.9203844, -13.3535213, 5.9115353, -15.3925705, 15.4330330
13: -9.2930412, 9.7554951, -9.2759132, 9.8178310, -16.3802338, 16.3099022
14: -86.1976852, -59.4851761, -86.0915375, -59.4853439, -19.9953003, 19.8467216
15: -29.5827732, -11.9112978, -29.5749855, -11.8944244, -12.1583023, 12.1427917
16: -43.4054260, -22.5511646, -43.3689995, -22.4840355, -16.3092499, 16.2193565
17: -100.0408020, -70.0146179, -99.9696426, -69.9536209, -22.1616974, 22.1436462
18: -17.7594337, 3.4588809, -17.8077259, 3.4569664, -13.6842995, 13.7295952
19: -21.0245590, -6.4486351, -21.0132980, -6.3954639, -12.4654732, 12.4116631
20: -8.1966372, 5.5854044, -8.2189064, 5.5836506, -13.7802877, 13.8043108
21: -30.4962425, -12.1475143, -30.4792824, -12.1036139, -16.1281357, 16.1001663
22: -24.8028851, -8.3607082, -24.7959576, -8.3476067, -12.1539917, 12.1717567
23: -16.8909721, 0.1535513, -16.8783150, 0.1545978, -14.1277008, 14.1518250
24: -8.0173664, 6.9047995, -8.0569763, 6.9005852, -12.7737770, 12.8336449
25: -4.5975971, 11.7316093, -4.6039181, 11.7310104, -14.1618347, 14.1745224
26: -23.0594044, -1.5649076, -23.1188850, -1.5743580, -18.2892075, 18.3901443
27: -17.8066597, -3.7849538, -17.8600063, -3.7915344, -12.8833008, 12.9522018
28: -3.3363328, 16.1720161, -3.3593037, 16.1603775, -15.9687195, 15.9887924
29: -41.7398453, -23.3517323, -41.7358932, -23.3121262, -14.5436172, 14.5655518
30: -11.7999220, 7.2652693, -11.8631058, 7.2527332, -17.7295837, 17.8233337
31: -22.9053993, -4.3977103, -22.8950806, -4.3473768, -15.3364868, 15.2358055
32: -3.7952366, 10.6095333, -3.7932734, 10.6029129, -11.2506866, 11.2400742
33: 10.5001202, 30.9262962, 10.4962254, 30.8872147, -16.2791748, 16.3129044
34: 11.2690716, 29.0317192, 11.1718054, 28.9816475, -11.3746986, 11.5680199
35: 22.9114571, 40.5390244, 22.8771210, 40.4887543, -11.3105698, 11.3619461
36: 17.9403095, 34.5604019, 17.8893585, 34.5261765, -12.3420486, 12.4385796
37: 7.8663945, 28.1338043, 7.8285484, 28.1257019, -16.7485962, 16.7760849
38: 6.5868602, 26.6040268, 6.5309572, 26.5814438, -14.3855591, 14.4729614
39: 5.7060528, 25.9491615, 5.7145801, 25.9534531, -16.2311401, 16.2205582
40: 0.5907502, 19.8931160, 0.5392790, 19.8640099, -12.5522957, 12.6673470
41: -4.0861945, 9.1111431, -4.0861254, 9.0941963, -10.9466438, 10.9910774
42: -27.5824146, -10.8346472, -27.5841522, -10.8392649, -11.5191994, 11.5412979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2038599, upper bound: 5.2126842
time: 33.17 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2126840, upper bound: 5.2126842
time: 29.03 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 64.30 seconds
IS_B1_B2_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2018698, upper bound: 5.1721453
IS_B1_B2_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2107114, upper bound: 5.1721453
IS_B1_B2_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2024406, upper bound: 5.1913972
IS_B1_B2_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2112808, upper bound: 5.1913972
IS_B2_A1_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2029229, upper bound: 5.1919209
IS_B2_A1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2117654, upper bound: 5.1919209
IS_B2_A2_A1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1713664, upper bound: 5.2011801
IS_B2_A2_A1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1713664, upper bound: 5.2100186
IS_B2_A2_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1880682, upper bound: 5.2033268
IS_B2_A2_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1880682, upper bound: 5.2121658
IS_B2_A2_A2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1879697, upper bound: 5.2019343
IS_B2_A2_A2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1879697, upper bound: 5.2107725
IS_B2_A2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1789051, upper bound: 5.2120410
IS_B2_A2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1858646, upper bound: 5.2120410
IS_B2_A2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1846425, upper bound: 5.2120410
IS_B2_A2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1934783, upper bound: 5.2120410
IS_B2_A2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.1981265, upper bound: 5.2126842
IS_B2_A2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2050768, upper bound: 5.2126842
IS_B2_A2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2038599, upper bound: 5.2126842
IS_B2_A2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 64.30
Output dim: 35, lower bound: -5.2126840, upper bound: 5.2126842

## BFS IS instance: IS_B2_A1_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -57.6655807, -32.6212616, -57.6335526, -32.5483017, -17.5820427, 17.4767952
1: -39.2070427, -20.2038574, -39.1809921, -20.1441936, -11.9331017, 11.8334351
2: -27.2439575, -11.1583157, -27.2230434, -11.1065025, -11.0728455, 10.9799652
3: -31.5712929, -14.0881710, -31.5693855, -14.0544443, -10.9707451, 10.9399033
4: -29.3858986, -8.6538658, -29.3652115, -8.5553293, -14.2432289, 14.1119385
5: -31.7555389, -13.5418434, -31.7444191, -13.4860172, -12.2067795, 12.1473656
6: -14.8850260, 2.9154253, -14.8906012, 2.8803029, -11.5571938, 11.6229858
7: -46.6696892, -25.5364227, -46.6327972, -25.4854279, -12.0783234, 11.9741402
8: -41.4618454, -19.8364525, -41.4254150, -19.8104153, -10.7439995, 10.6618805
9: -24.2606850, -5.1505804, -24.2462482, -5.0525942, -16.5072174, 16.3902359
10: -52.0866852, -29.6834412, -52.0394707, -29.5191174, -17.2117004, 16.9877930
11: -47.9175797, -27.1110382, -47.9055824, -27.0900688, -14.9862137, 15.1115189
12: -13.3053865, 5.8923068, -13.3263226, 5.9320641, -15.4079018, 15.3795509
13: -9.2852650, 9.7426853, -9.2769623, 9.8565960, -16.4276199, 16.2665176
14: -86.1296463, -59.5370636, -86.0563354, -59.4937210, -19.9040298, 19.8532829
15: -29.5606880, -11.9289455, -29.5600548, -11.8822050, -12.1520920, 12.1081009
16: -43.3829498, -22.5629768, -43.3580856, -22.4378166, -16.3213043, 16.1999512
17: -100.0035706, -70.0318222, -99.9513245, -69.9359283, -22.0587006, 22.1713791
18: -17.7316227, 3.4375706, -17.8091431, 3.4510727, -13.6302681, 13.7663002
19: -21.0064220, -6.4649329, -21.0051651, -6.3785696, -12.4465599, 12.3798714
20: -8.1849117, 5.5784621, -8.2247696, 5.5779581, -13.7628698, 13.8032322
21: -30.4692345, -12.1773806, -30.4750080, -12.1004143, -16.0672989, 16.0860596
22: -24.7915344, -8.3620548, -24.7858086, -8.3322611, -12.1368256, 12.1660156
23: -16.8608246, 0.1156094, -16.8913269, 0.1292705, -14.0622025, 14.1509361
24: -8.0100040, 6.8981013, -8.0933132, 6.8971958, -12.7324905, 12.8580093
25: -4.5680208, 11.6989126, -4.6099749, 11.7115765, -14.0949097, 14.1639175
26: -23.0303192, -1.5779564, -23.1148949, -1.5805902, -18.2336197, 18.4016190
27: -17.7995415, -3.7877717, -17.8995781, -3.7937031, -12.8550034, 12.9955597
28: -3.3157434, 16.1473885, -3.3825235, 16.1415539, -15.9254150, 15.9836044
29: -41.7277374, -23.3633232, -41.7334328, -23.3037434, -14.5167465, 14.5704880
30: -11.7776299, 7.2358751, -11.9210014, 7.2362032, -17.6793823, 17.8484268
31: -22.8900394, -4.3984833, -22.8846512, -4.3210073, -15.2961807, 15.2161980
32: -3.7567019, 10.5790300, -3.7740040, 10.6237144, -11.2746658, 11.1796761
33: 10.5301437, 30.8975544, 10.4911842, 30.8744907, -16.2856293, 16.2933731
34: 11.2743692, 29.0248051, 11.1076851, 28.9751205, -11.3507347, 11.6322289
35: 22.9507103, 40.5017395, 22.8391304, 40.4652328, -11.2493744, 11.4148140
36: 17.9469757, 34.5534592, 17.8591042, 34.5203056, -12.3264771, 12.4581337
37: 7.9116430, 28.0775223, 7.8175397, 28.0906868, -16.7106628, 16.7724609
38: 6.5957041, 26.5933647, 6.4963837, 26.5729141, -14.3633423, 14.5052719
39: 5.7211556, 25.9489346, 5.7197151, 25.9851799, -16.2551956, 16.1975708
40: 0.6182060, 19.8724709, 0.5321746, 19.8626442, -12.5733643, 12.6190376
41: -4.0844774, 9.1085825, -4.0877295, 9.0985327, -10.9585381, 10.9481354
42: -27.5798836, -10.8380737, -27.5836735, -10.8338022, -11.5218582, 11.5215874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 755

## Relational analysis of IS_B2_A1_A2_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1951654, upper bound: 5.1914334
time: 7.17 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2112797, upper bound: 5.1914367
time: 6.03 seconds

## BFS IS instance: IS_B2_A2_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -57.6364594, -32.5767746, -57.6576004, -32.6295815, -17.4584351, 17.5550308
1: -39.1845551, -20.1671085, -39.2010498, -20.2103462, -11.8353043, 11.9110985
2: -27.2249794, -11.1216373, -27.2407112, -11.1632881, -10.9770393, 11.0531921
3: -31.5677719, -14.0621738, -31.5726852, -14.0884094, -10.9316063, 10.9583244
4: -29.3791199, -8.5728092, -29.4017639, -8.6387119, -14.1219025, 14.2377586
5: -31.7433357, -13.5027037, -31.7502880, -13.5444984, -12.1448441, 12.1807671
6: -14.8683176, 2.8794937, -14.8829346, 2.8986831, -11.5814819, 11.5638657
7: -46.6363297, -25.5150490, -46.6563759, -25.5487213, -11.9596405, 12.0387230
8: -41.4295235, -19.8484631, -41.4510803, -19.8520947, -10.6365128, 10.7053337
9: -24.2598381, -5.0636454, -24.2769089, -5.1340833, -16.4266968, 16.5102768
10: -52.0650024, -29.5554276, -52.1047745, -29.6616592, -16.9776917, 17.2148438
11: -47.9058304, -27.0979652, -47.9133682, -27.1044960, -15.1112213, 15.0302238
12: -13.3364401, 5.9274225, -13.3357811, 5.9141278, -15.3707275, 15.4220009
13: -9.2604342, 9.8424616, -9.2718220, 9.7465048, -16.2547989, 16.4065399
14: -86.0864868, -59.5530701, -86.1343765, -59.5232735, -19.8647614, 19.8298531
15: -29.5663929, -11.8964167, -29.5737896, -11.9207067, -12.1240616, 12.1425705
16: -43.3640442, -22.4651375, -43.3758278, -22.5676861, -16.1990967, 16.2996368
17: -99.9657745, -69.9887085, -99.9931488, -70.0457764, -22.1364975, 22.0604019
18: -17.8174667, 3.4521616, -17.7564125, 3.4552517, -13.7761650, 13.6421738
19: -21.0004616, -6.3844357, -21.0073833, -6.4520760, -12.3758049, 12.4877510
20: -8.2201748, 5.5720205, -8.1902294, 5.5777750, -13.7979498, 13.7622499
21: -30.4734383, -12.1002226, -30.4771996, -12.1531553, -16.0866547, 16.1225090
22: -24.7811508, -8.3407154, -24.7880917, -8.3639393, -12.1485748, 12.1598396
23: -16.8885727, 0.1322005, -16.8769493, 0.1415939, -14.1489563, 14.1116867
24: -8.0878410, 6.8960543, -8.0121651, 6.9018202, -12.8591995, 12.7560120
25: -4.6035843, 11.7213926, -4.5885682, 11.7254124, -14.1758118, 14.1382256
26: -23.1171913, -1.5831451, -23.0511856, -1.5764642, -18.3928299, 18.2478561
27: -17.8934898, -3.7960711, -17.8042946, -3.7891483, -12.9876556, 12.8568916
28: -3.3740854, 16.1510239, -3.3281379, 16.1656799, -15.9919662, 15.9464874
29: -41.7311325, -23.3078041, -41.7312775, -23.3567162, -14.5611191, 14.5319252
30: -11.9193668, 7.2459335, -11.7954874, 7.2584677, -17.8621979, 17.7092896
31: -22.8749008, -4.3345499, -22.8900032, -4.3982363, -15.2223587, 15.3261185
32: -3.7753286, 10.6242189, -3.7829866, 10.6021442, -11.1861725, 11.2960930
33: 10.5325413, 30.8887558, 10.5279808, 30.9052811, -16.2479858, 16.2679443
34: 11.1469002, 28.9750977, 11.2871151, 29.0004044, -11.5749283, 11.3422928
35: 22.8818588, 40.4851952, 22.9380951, 40.5128784, -11.3900871, 11.2291718
36: 17.9045601, 34.5222244, 17.9684258, 34.5416832, -12.4083481, 12.3097992
37: 7.8265190, 28.1252670, 7.8753934, 28.1318665, -16.7822647, 16.7154236
38: 6.5424309, 26.5724564, 6.6103926, 26.5839386, -14.4625549, 14.3405113
39: 5.7566738, 25.9833641, 5.7393513, 25.9472237, -16.1628571, 16.2372437
40: 0.5431299, 19.8604183, 0.6043320, 19.8745003, -12.5909004, 12.5775452
41: -4.0727487, 9.0987120, -4.0789938, 9.1004095, -10.9361534, 10.9750595
42: -27.5761051, -10.8372059, -27.5804787, -10.8438110, -11.5028191, 11.5363388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=114, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 755

## Relational analysis of IS_B2_A2_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1875283, upper bound: 5.1954958
time: 9.56 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_A2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1875553, upper bound: 5.2116577
time: 9.23 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -57.6425247, -32.6377831, -57.6054573, -32.6293221, -17.4861107, 17.4499130
1: -39.1880646, -20.2181721, -39.1608810, -20.2109070, -11.8554382, 11.8281136
2: -27.2237816, -11.1716614, -27.2006931, -11.1667566, -10.9888268, 10.9621162
3: -31.5621948, -14.0888176, -31.5517731, -14.0878687, -10.9260521, 10.9179459
4: -29.3772564, -8.6493721, -29.3466377, -8.6432705, -14.1382446, 14.1204834
5: -31.7397614, -13.5486031, -31.7205849, -13.5439816, -12.1420174, 12.1175385
6: -14.8705359, 2.8974886, -14.8825598, 2.8774548, -11.5428848, 11.5934067
7: -46.6448555, -25.5517902, -46.6069336, -25.5445328, -11.9930611, 11.9490585
8: -41.4443626, -19.8614750, -41.4208298, -19.8498878, -10.6710854, 10.6461945
9: -24.2644634, -5.1394181, -24.2337227, -5.1367497, -16.4432907, 16.4038696
10: -52.0838013, -29.6730900, -52.0221596, -29.6662102, -17.0199280, 17.0120163
11: -47.9114304, -27.0944157, -47.8919830, -27.0946503, -15.0125504, 15.0386124
12: -13.3411465, 5.9055448, -13.3314047, 5.8992810, -15.3674812, 15.3918839
13: -9.2428417, 9.7316914, -9.2110510, 9.7388515, -16.2546768, 16.2422447
14: -86.1350937, -59.5261803, -86.0700684, -59.5103989, -19.8910217, 19.7765808
15: -29.5605659, -11.9302673, -29.5486698, -11.9251337, -12.1260033, 12.1090355
16: -43.3629303, -22.5689335, -43.3105202, -22.5634117, -16.1931610, 16.1557732
17: -99.9677658, -70.0445251, -99.8853683, -70.0319824, -22.0785828, 22.0362701
18: -17.7472763, 3.4328370, -17.7452507, 3.4114537, -13.6362076, 13.6416283
19: -20.9999619, -6.4470730, -20.9741058, -6.4433117, -12.3774757, 12.3754120
20: -8.1822290, 5.5753045, -8.1806860, 5.5654788, -13.7477074, 13.7559910
21: -30.4682770, -12.1465740, -30.4359608, -12.1447239, -16.0497894, 16.0560913
22: -24.7899971, -8.3610878, -24.7723808, -8.3604555, -12.1236115, 12.1283340
23: -16.8810177, 0.1511515, -16.8694191, 0.1473722, -14.1095276, 14.1228256
24: -8.0007906, 6.8785591, -8.0035629, 6.8585510, -12.7291946, 12.7501221
25: -4.5836115, 11.7247076, -4.5796437, 11.7190914, -14.1396255, 14.1499557
26: -23.0409393, -1.5827804, -23.0383625, -1.6010337, -18.2488327, 18.2617416
27: -17.7899780, -3.8162115, -17.7929287, -3.8409739, -12.8338089, 12.8529167
28: -3.3147964, 16.1520424, -3.3159804, 16.1389980, -15.9338455, 15.9436264
29: -41.7172050, -23.3551884, -41.6969147, -23.3550434, -14.5019226, 14.5127182
30: -11.7805204, 7.2150998, -11.7797518, 7.1831355, -17.6549606, 17.6920090
31: -22.8789024, -4.3961000, -22.8617420, -4.3916798, -15.2291183, 15.1994705
32: -3.7786026, 10.5989304, -3.7771592, 10.5872221, -11.1803093, 11.2047920
33: 10.5392332, 30.9030800, 10.5295782, 30.8828468, -16.2225113, 16.2525330
34: 11.3003445, 28.9640694, 11.2879524, 28.8976383, -11.2838249, 11.3675613
35: 22.9502335, 40.4957008, 22.9393463, 40.4557304, -11.2568474, 11.2615738
36: 17.9743958, 34.5257263, 17.9629517, 34.4915123, -12.2907677, 12.3372231
37: 7.8834043, 28.1148510, 7.8825288, 28.1012573, -16.7170486, 16.7077179
38: 6.6149530, 26.5761013, 6.6017380, 26.5377808, -14.3225670, 14.3630371
39: 5.7382617, 25.9459610, 5.7425551, 25.9426823, -16.1770554, 16.1817551
40: 0.6082258, 19.8580303, 0.6018305, 19.8240013, -12.5130424, 12.5733223
41: -4.0703735, 9.0922756, -4.0721760, 9.0787382, -10.9244194, 10.9601059
42: -27.5674915, -10.8511667, -27.5653725, -10.8603477, -11.4901543, 11.5089760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1642691, upper bound: 5.2113685
time: 12.77 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1785456, upper bound: 5.2116833
time: 6.22 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -57.6468468, -32.6377907, -57.6167412, -32.6114426, -17.4993210, 17.4566193
1: -39.1921844, -20.2185364, -39.1701279, -20.1941624, -11.8720131, 11.8318443
2: -27.2249241, -11.1714888, -27.2042141, -11.1583633, -10.9986267, 10.9643211
3: -31.5627022, -14.0923634, -31.5530720, -14.0893688, -10.9296341, 10.9245071
4: -29.3813229, -8.6491098, -29.3561764, -8.6129436, -14.1698685, 14.1277237
5: -31.7399368, -13.5484819, -31.7222023, -13.5365467, -12.1427650, 12.1330147
6: -14.8704147, 2.8969049, -14.8822613, 2.8776493, -11.5443230, 11.5948219
7: -46.6474876, -25.5525646, -46.6132431, -25.5409298, -11.9975891, 11.9517860
8: -41.4443550, -19.8616905, -41.4210930, -19.8470001, -10.6831055, 10.6429405
9: -24.2718582, -5.1393919, -24.2523518, -5.0904574, -16.4916382, 16.4166412
10: -52.0936050, -29.6724205, -52.0433884, -29.6028328, -17.0879135, 17.0249252
11: -47.9126587, -27.1016808, -47.8896561, -27.1047001, -15.0099411, 15.0694733
12: -13.3435154, 5.9072418, -13.3431473, 5.9255319, -15.3961182, 15.4046631
13: -9.2516994, 9.7321119, -9.2346230, 9.7821589, -16.3038635, 16.2562904
14: -86.1367645, -59.5284348, -86.0784760, -59.5140457, -19.8846817, 19.8138847
15: -29.5615044, -11.9298573, -29.5516872, -11.9108152, -12.1362457, 12.1091461
16: -43.3731232, -22.5698853, -43.3360786, -22.5172329, -16.2310410, 16.1719322
17: -99.9788971, -70.0452881, -99.9117737, -70.0132675, -22.0782776, 22.0783463
18: -17.7485523, 3.4347878, -17.7644672, 3.4185743, -13.6421661, 13.6851311
19: -21.0029182, -6.4471560, -20.9813023, -6.4195409, -12.4019241, 12.3787003
20: -8.1825733, 5.5762992, -8.1925335, 5.5693951, -13.7519684, 13.7688332
21: -30.4739952, -12.1466169, -30.4490738, -12.1253977, -16.0561066, 16.0773888
22: -24.7910805, -8.3610506, -24.7766418, -8.3444223, -12.1386375, 12.1354790
23: -16.8815708, 0.1523674, -16.8857269, 0.1520780, -14.1122284, 14.1459198
24: -8.0010900, 6.8874054, -8.0422802, 6.8775363, -12.7419090, 12.7907410
25: -4.5841503, 11.7254734, -4.5890007, 11.7222414, -14.1371155, 14.1737099
26: -23.0407925, -1.5839105, -23.0510273, -1.5992947, -18.2555466, 18.2803574
27: -17.7906952, -3.8077602, -17.8343716, -3.8223262, -12.8467941, 12.8943367
28: -3.3153245, 16.1538601, -3.3422940, 16.1442413, -15.9391861, 15.9672928
29: -41.7200623, -23.3550262, -41.7034683, -23.3395519, -14.4943237, 14.5306320
30: -11.7818584, 7.2271719, -11.8406582, 7.2105970, -17.6736908, 17.7521286
31: -22.8812790, -4.3963051, -22.8679390, -4.3671408, -15.2588425, 15.2045097
32: -3.7831054, 10.6010399, -3.7903180, 10.6152086, -11.2239037, 11.2161903
33: 10.5390320, 30.9028282, 10.5191488, 30.8870144, -16.2460556, 16.2573700
34: 11.2995062, 28.9766712, 11.2251415, 28.9232903, -11.3007050, 11.4572792
35: 22.9495621, 40.5013847, 22.8997726, 40.4678116, -11.2628250, 11.3068352
36: 17.9753647, 34.5309601, 17.9336662, 34.5019035, -12.2942429, 12.3659019
37: 7.8834944, 28.1158352, 7.8666844, 28.1039886, -16.7243881, 16.7163467
38: 6.6153955, 26.5784950, 6.5693960, 26.5483875, -14.3316383, 14.4056282
39: 5.7372904, 25.9463711, 5.7279024, 25.9752464, -16.2144318, 16.1954346
40: 0.6079645, 19.8609791, 0.5787215, 19.8306637, -12.5426254, 12.5703545
41: -4.0722995, 9.0938816, -4.0782318, 9.0893688, -10.9479332, 10.9612656
42: -27.5700798, -10.8501606, -27.5706215, -10.8503141, -11.5103798, 11.5076141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1712411, upper bound: 5.2113685
time: 16.77 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1855060, upper bound: 5.2116833
time: 13.36 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -57.6531258, -32.6365433, -57.6259384, -32.5778008, -17.5437050, 17.4597054
1: -39.1955414, -20.2171173, -39.1755905, -20.1710396, -11.8966637, 11.8317528
2: -27.2350254, -11.1708946, -27.2222099, -11.1230831, -11.0428772, 10.9714851
3: -31.5713692, -14.0886488, -31.5694885, -14.0574312, -10.9593964, 10.9241142
4: -29.3882217, -8.6482925, -29.3688202, -8.5977192, -14.1775055, 14.1295090
5: -31.7512283, -13.5481443, -31.7427921, -13.4998703, -12.1939697, 12.1239967
6: -14.8705759, 2.8959904, -14.8826189, 2.8797331, -11.5487785, 11.5953598
7: -46.6573448, -25.5512505, -46.6309853, -25.4997864, -12.0501175, 11.9583740
8: -41.4486465, -19.8608875, -41.4298172, -19.8308811, -10.6923218, 10.6509514
9: -24.2662296, -5.1387930, -24.2402439, -5.1073017, -16.4617691, 16.4061432
10: -52.0930138, -29.6710072, -52.0428200, -29.6020718, -17.0846863, 17.0220490
11: -47.9189072, -27.0941734, -47.9079361, -27.0748463, -15.0249634, 15.0711975
12: -13.3354959, 5.9090204, -13.3293209, 5.9051805, -15.3660164, 15.3935890
13: -9.2576427, 9.7331333, -9.2451954, 9.8051186, -16.3320389, 16.2604256
14: -86.1383514, -59.5275803, -86.0810471, -59.5113220, -19.8935394, 19.8005066
15: -29.5650234, -11.9291468, -29.5655060, -11.9046946, -12.1306229, 12.1194458
16: -43.3742905, -22.5681458, -43.3368073, -22.4939995, -16.2733536, 16.1859283
17: -99.9930878, -70.0444183, -99.9409637, -69.9703369, -22.1172714, 22.0985794
18: -17.7494736, 3.4487565, -17.8008194, 3.4431834, -13.6556320, 13.7015686
19: -21.0085316, -6.4470429, -20.9940834, -6.3944421, -12.4498672, 12.3863144
20: -8.1845398, 5.5783978, -8.2116365, 5.5735664, -13.7581062, 13.7900343
21: -30.4788990, -12.1464386, -30.4604931, -12.1035490, -16.1076508, 16.0675735
22: -24.7883358, -8.3610640, -24.7800598, -8.3481026, -12.1373863, 12.1431656
23: -16.8828392, 0.1488835, -16.8732910, 0.1452382, -14.1101379, 14.1291733
24: -8.0020399, 6.8887186, -8.0481787, 6.8798261, -12.7434082, 12.7993202
25: -4.5856204, 11.7275085, -4.5967669, 11.7259359, -14.1371460, 14.1506348
26: -23.0441093, -1.5766029, -23.1084595, -1.5862803, -18.2665558, 18.3447342
27: -17.7916870, -3.8040276, -17.8511543, -3.8153496, -12.8514023, 12.9159660
28: -3.3170018, 16.1537113, -3.3480716, 16.1438656, -15.9340973, 15.9597855
29: -41.7293091, -23.3552475, -41.7254333, -23.3138046, -14.5235214, 14.5420914
30: -11.7830648, 7.2318430, -11.8526077, 7.2175684, -17.6826935, 17.7754517
31: -22.8834839, -4.3958592, -22.8726501, -4.3467226, -15.3098831, 15.2072449
32: -3.7749591, 10.6023026, -3.7732182, 10.5969086, -11.2283020, 11.2090950
33: 10.5376682, 30.9034061, 10.5194950, 30.8858871, -16.2402420, 16.2578812
34: 11.2987547, 28.9862785, 11.1890564, 28.9455719, -11.3126221, 11.5050354
35: 22.9487190, 40.5042152, 22.8992710, 40.4719238, -11.2665939, 11.3072281
36: 17.9738655, 34.5349464, 17.9087486, 34.5110779, -12.3022003, 12.3955345
37: 7.8827643, 28.1243401, 7.8400497, 28.1195965, -16.7251434, 16.7516594
38: 6.6149001, 26.5840111, 6.5466433, 26.5591660, -14.3364105, 14.4376297
39: 5.7450590, 25.9470634, 5.7500157, 25.9530277, -16.1898346, 16.1789856
40: 0.6083317, 19.8727188, 0.5504541, 19.8546867, -12.5266838, 12.6279030
41: -4.0705638, 9.0951576, -4.0740576, 9.0886536, -10.9262810, 10.9579048
42: -27.5716286, -10.8488588, -27.5746231, -10.8447933, -11.5038490, 11.5124969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1700128, upper bound: 5.2113685
time: 16.75 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1842833, upper bound: 5.2116833
time: 5.71 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -57.6575813, -32.6364861, -57.6395111, -32.5597382, -17.5580978, 17.4690170
1: -39.1997414, -20.2174110, -39.1856346, -20.1541977, -11.9152107, 11.8365211
2: -27.2362194, -11.1706629, -27.2262192, -11.1142921, -11.0537224, 10.9743347
3: -31.5719681, -14.0920563, -31.5710564, -14.0578451, -10.9630928, 10.9310951
4: -29.3928108, -8.6479855, -29.3807793, -8.5646973, -14.2144394, 14.1379242
5: -31.7514019, -13.5479784, -31.7444611, -13.4917202, -12.1970367, 12.1396599
6: -14.8704739, 2.8955302, -14.8830547, 2.8802881, -11.5506973, 11.5977211
7: -46.6600113, -25.5520935, -46.6374741, -25.4960308, -12.0548210, 11.9612923
8: -41.4486504, -19.8610725, -41.4301529, -19.8262520, -10.7065201, 10.6478214
9: -24.2741203, -5.1386385, -24.2618313, -5.0577526, -16.5155640, 16.4210205
10: -52.1028214, -29.6701279, -52.0660782, -29.5330887, -17.1627655, 17.0357361
11: -47.9202690, -27.1013432, -47.9064865, -27.0845299, -15.0238724, 15.1062355
12: -13.3380051, 5.9109955, -13.3430395, 5.9335990, -15.3972511, 15.4077301
13: -9.2674065, 9.7336435, -9.2723761, 9.8504486, -16.3868561, 16.2777367
14: -86.1405029, -59.5298386, -86.0912933, -59.5149956, -19.8909454, 19.8411636
15: -29.5664845, -11.9287081, -29.5696602, -11.8896818, -12.1436501, 12.1202965
16: -43.3861313, -22.5690422, -43.3665466, -22.4452057, -16.3146515, 16.2048645
17: -100.0047760, -70.0451584, -99.9687195, -69.9511642, -22.1225128, 22.1437836
18: -17.7510338, 3.4507928, -17.8210449, 3.4509728, -13.6641312, 13.7491646
19: -21.0121307, -6.4471111, -21.0033150, -6.3680706, -12.4757271, 12.3904991
20: -8.1850185, 5.5799198, -8.2240467, 5.5793834, -13.7644024, 13.8039665
21: -30.4848652, -12.1464081, -30.4753590, -12.0829363, -16.1151886, 16.0907936
22: -24.7895908, -8.3610134, -24.7847729, -8.3319225, -12.1570969, 12.1514435
23: -16.8834724, 0.1501507, -16.8903694, 0.1504617, -14.1133499, 14.1543159
24: -8.0024490, 6.8976212, -8.0885983, 6.8990707, -12.7580719, 12.8438377
25: -4.5863409, 11.7283726, -4.6070013, 11.7293186, -14.1353378, 14.1766968
26: -23.0441551, -1.5767069, -23.1216717, -1.5809636, -18.2752686, 18.3639069
27: -17.7925835, -3.7942650, -17.8957672, -3.7938604, -12.8661728, 12.9626503
28: -3.3176272, 16.1564198, -3.3756475, 16.1516914, -15.9404297, 15.9855957
29: -41.7322159, -23.3550377, -41.7327805, -23.2979298, -14.5234985, 14.5621414
30: -11.7845984, 7.2455730, -11.9164543, 7.2494164, -17.7061996, 17.8433228
31: -22.8858490, -4.3960156, -22.8808289, -4.3198237, -15.3398056, 15.2135391
32: -3.7799239, 10.6047859, -3.7886004, 10.6259212, -11.2730827, 11.2227707
33: 10.5372849, 30.9031944, 10.5065928, 30.8901062, -16.2645798, 16.2650604
34: 11.2977848, 28.9998684, 11.1204987, 28.9758568, -11.3323174, 11.5979156
35: 22.9478416, 40.5106163, 22.8557053, 40.4858017, -11.2731285, 11.3580017
36: 17.9747715, 34.5404510, 17.8759384, 34.5226173, -12.3066711, 12.4292793
37: 7.8825841, 28.1253014, 7.8222594, 28.1223564, -16.7326355, 16.7648697
38: 6.6152620, 26.5880737, 6.5091991, 26.5734882, -14.3472023, 14.4886475
39: 5.7438383, 25.9475613, 5.7320561, 25.9857330, -16.2298813, 16.1960297
40: 0.6079092, 19.8757858, 0.5265884, 19.8620186, -12.5581894, 12.6325645
41: -4.0725651, 9.0970268, -4.0811214, 9.0998745, -10.9526443, 10.9614067
42: -27.5742245, -10.8476467, -27.5798702, -10.8340549, -11.5252304, 11.5116730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1788565, upper bound: 5.2113685
time: 5.63 seconds

## Relational analysis of IS_B2_A2_A2_A2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1931200, upper bound: 5.2116833
time: 45.92 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -57.6677322, -32.6145210, -57.6084824, -32.6170120, -17.5241394, 17.4702530
1: -39.2055359, -20.1980877, -39.1616898, -20.2000751, -11.8834953, 11.8442268
2: -27.2389297, -11.1557045, -27.2013817, -11.1586390, -11.0121384, 10.9754562
3: -31.5642262, -14.0801172, -31.5511131, -14.0834694, -10.9356995, 10.9282150
4: -29.3984928, -8.6294785, -29.3471088, -8.6329384, -14.1698761, 14.1353188
5: -31.7463894, -13.5364084, -31.7209530, -13.5373926, -12.1551704, 12.1297913
6: -14.8916674, 2.9179893, -14.8934937, 2.8779960, -11.5526047, 11.6261940
7: -46.6641617, -25.5317612, -46.6072998, -25.5335388, -12.0232620, 11.9642525
8: -41.4665680, -19.8303108, -41.4211998, -19.8333969, -10.7096024, 10.6692219
9: -24.2782555, -5.1267853, -24.2339706, -5.1302633, -16.4624863, 16.4130096
10: -52.1132431, -29.6410675, -52.0223885, -29.6502819, -17.0664062, 17.0349808
11: -47.9171944, -27.0928116, -47.8930969, -27.0940800, -15.0131836, 15.0542603
12: -13.3436031, 5.9139009, -13.3339081, 5.8998842, -15.3788834, 15.4075165
13: -9.2651081, 9.7528343, -9.2164822, 9.7492580, -16.2877045, 16.2681274
14: -86.1897964, -59.4861145, -86.0719070, -59.4885063, -19.9662704, 19.8015823
15: -29.5740948, -11.9131851, -29.5499287, -11.9163523, -12.1484375, 12.1242447
16: -43.3784409, -22.5526600, -43.3122215, -22.5547714, -16.2141342, 16.1695137
17: -100.0008850, -70.0152054, -99.8861694, -70.0162201, -22.0888977, 22.0513229
18: -17.7548523, 3.4360385, -17.7476521, 3.4121866, -13.6415024, 13.6548004
19: -21.0076046, -6.4487696, -20.9772606, -6.4446087, -12.3817177, 12.3833084
20: -8.1925793, 5.5774646, -8.1846724, 5.5658751, -13.7584543, 13.7621365
21: -30.4769173, -12.1481876, -30.4381924, -12.1458569, -16.0527344, 16.0719757
22: -24.7976017, -8.3610840, -24.7750473, -8.3606529, -12.1289825, 12.1429939
23: -16.8878746, 0.1504978, -16.8719215, 0.1464595, -14.1121521, 14.1351852
24: -8.0150700, 6.8840351, -8.0103960, 6.8588743, -12.7404556, 12.7730141
25: -4.5941067, 11.7262239, -4.5840583, 11.7192001, -14.1490555, 14.1628609
26: -23.0535526, -1.5765984, -23.0435982, -1.5997486, -18.2560196, 18.2953796
27: -17.8032246, -3.8088915, -17.7985153, -3.8400836, -12.8439560, 12.8762474
28: -3.3325498, 16.1625938, -3.3241644, 16.1402512, -15.9525146, 15.9623184
29: -41.7232552, -23.3521347, -41.6987724, -23.3543816, -14.5105286, 14.5257759
30: -11.7952271, 7.2307053, -11.7862663, 7.1840014, -17.6705017, 17.7223129
31: -22.8920784, -4.3985076, -22.8676109, -4.3933997, -15.2409286, 15.2068939
32: -3.7917795, 10.6028547, -3.7835145, 10.5870066, -11.1886940, 11.2148972
33: 10.5043764, 30.9255161, 10.5115671, 30.8833008, -16.2509232, 16.2925034
34: 11.2722254, 28.9906540, 11.2737074, 28.8981113, -11.3073044, 11.4078331
35: 22.9149513, 40.5217628, 22.9210682, 40.4557724, -11.2846184, 11.3050423
36: 17.9419823, 34.5433197, 17.9457989, 34.4916611, -12.3164825, 12.3712959
37: 7.8693118, 28.1212006, 7.8754873, 28.1015034, -16.7298431, 16.7201424
38: 6.5875282, 26.5848827, 6.5872364, 26.5384560, -14.3487968, 14.3850937
39: 5.7103281, 25.9469643, 5.7283220, 25.9410706, -16.2046204, 16.1994705
40: 0.5918856, 19.8741856, 0.5930557, 19.8252068, -12.5246582, 12.5959206
41: -4.0828710, 9.1056366, -4.0782270, 9.0793447, -10.9333649, 10.9788399
42: -27.5753498, -10.8391552, -27.5691891, -10.8589849, -11.4955635, 11.5251846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1452
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 755

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1815266, upper bound: 5.2121951
time: 13.73 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1976408, upper bound: 5.2121951
time: 8.79 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -57.6720085, -32.6145706, -57.6197128, -32.5991058, -17.5373611, 17.4769402
1: -39.2096672, -20.1984272, -39.1709442, -20.1833763, -11.9000893, 11.8479385
2: -27.2401047, -11.1555233, -27.2048969, -11.1502571, -11.0219345, 10.9776573
3: -31.5647240, -14.0836134, -31.5524025, -14.0849113, -10.9392815, 10.9347878
4: -29.4025536, -8.6292343, -29.3566360, -8.6026306, -14.2015266, 14.1426010
5: -31.7465572, -13.5362949, -31.7225590, -13.5299816, -12.1559067, 12.1452599
6: -14.8915615, 2.9174471, -14.8932199, 2.8781862, -11.5540657, 11.6276321
7: -46.6667671, -25.5325871, -46.6136322, -25.5299950, -12.0277672, 11.9669914
8: -41.4665222, -19.8305206, -41.4214325, -19.8304749, -10.7216415, 10.6659775
9: -24.2856750, -5.1267042, -24.2525673, -5.0839496, -16.5108185, 16.4257812
10: -52.1230125, -29.6403828, -52.0436172, -29.5869465, -17.1344070, 17.0478668
11: -47.9183578, -27.1000671, -47.8907394, -27.1041336, -15.0105438, 15.0851593
12: -13.3459549, 5.9155960, -13.3456364, 5.9261417, -15.4074821, 15.4203224
13: -9.2739792, 9.7532454, -9.2400761, 9.7925224, -16.3369370, 16.2821655
14: -86.1915207, -59.4883347, -86.0803223, -59.4921341, -19.9599457, 19.8388748
15: -29.5750351, -11.9127855, -29.5529366, -11.9020195, -12.1586952, 12.1243286
16: -43.3886299, -22.5536003, -43.3377609, -22.5086060, -16.2520447, 16.1856804
17: -100.0120697, -70.0159225, -99.9126511, -69.9974442, -22.0886612, 22.0933914
18: -17.7561264, 3.4379783, -17.7668839, 3.4192851, -13.6474609, 13.6983032
19: -21.0105877, -6.4488592, -20.9844646, -6.4208107, -12.4061470, 12.3866043
20: -8.1929045, 5.5784502, -8.1965284, 5.5697865, -13.7626915, 13.7749786
21: -30.4826889, -12.1482449, -30.4513454, -12.1264992, -16.0590286, 16.0932617
22: -24.7986984, -8.3610382, -24.7793064, -8.3445950, -12.1440506, 12.1501236
23: -16.8884239, 0.1517010, -16.8882446, 0.1511242, -14.1148300, 14.1582565
24: -8.0153809, 6.8928509, -8.0491238, 6.8778439, -12.7531586, 12.8136253
25: -4.5946808, 11.7269917, -4.5934329, 11.7223644, -14.1465378, 14.1865768
26: -23.0534363, -1.5777438, -23.0562630, -1.5980198, -18.2627258, 18.3139801
27: -17.8039818, -3.8004606, -17.8399696, -3.8214226, -12.8569641, 12.9176865
28: -3.3330705, 16.1644020, -3.3505054, 16.1455345, -15.9578400, 15.9859161
29: -41.7260780, -23.3520012, -41.7053413, -23.3389282, -14.5029449, 14.5436783
30: -11.7965765, 7.2427554, -11.8471603, 7.2114363, -17.6892166, 17.7824249
31: -22.8944397, -4.3987255, -22.8738174, -4.3688812, -15.2706757, 15.2119255
32: -3.7962570, 10.6049585, -3.7966762, 10.6149845, -11.2322502, 11.2263298
33: 10.5042210, 30.9253349, 10.5011234, 30.8874359, -16.2744675, 16.2973328
34: 11.2714100, 29.0032673, 11.2109375, 28.9237652, -11.3242226, 11.4975624
35: 22.9142799, 40.5274467, 22.8815174, 40.4678650, -11.2905807, 11.3502998
36: 17.9429932, 34.5485420, 17.9164772, 34.5020180, -12.3199921, 12.3999825
37: 7.8693857, 28.1221504, 7.8596101, 28.1042709, -16.7371597, 16.7287750
38: 6.5880022, 26.5872402, 6.5549011, 26.5491180, -14.3578644, 14.4277153
39: 5.7093158, 25.9473534, 5.7137008, 25.9736099, -16.2420197, 16.2131958
40: 0.5916324, 19.8771477, 0.5699887, 19.8318291, -12.5542450, 12.5929489
41: -4.0848093, 9.1072769, -4.0843019, 9.0899868, -10.9568787, 10.9800377
42: -27.5779457, -10.8381710, -27.5744858, -10.8489666, -11.5158081, 11.5238686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 755

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1884794, upper bound: 5.2121951
time: 29.83 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2045883, upper bound: 5.2121951
time: 6.31 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -57.6782913, -32.6132660, -57.6289482, -32.5654793, -17.5817261, 17.4800377
1: -39.2130127, -20.1970005, -39.1764030, -20.1602268, -11.9247398, 11.8478546
2: -27.2501564, -11.1549301, -27.2229385, -11.1149464, -11.0662003, 10.9848480
3: -31.5733604, -14.0798912, -31.5688133, -14.0530062, -10.9690437, 10.9343872
4: -29.4094563, -8.6284218, -29.3692703, -8.5874472, -14.2091522, 14.1443596
5: -31.7578392, -13.5359774, -31.7431717, -13.4932499, -12.2071228, 12.1362457
6: -14.8917217, 2.9164906, -14.8935575, 2.8802805, -11.5585022, 11.6281548
7: -46.6766434, -25.5313129, -46.6313591, -25.4888115, -12.0803185, 11.9735680
8: -41.4708519, -19.8297253, -41.4301338, -19.8143845, -10.7308464, 10.6739788
9: -24.2800140, -5.1261601, -24.2404900, -5.1008348, -16.4809418, 16.4152756
10: -52.1224365, -29.6390076, -52.0430527, -29.5861664, -17.1311874, 17.0449753
11: -47.9245834, -27.0925941, -47.9090385, -27.0742531, -15.0255737, 15.0868568
12: -13.3379517, 5.9173651, -13.3317957, 5.9057736, -15.3774033, 15.4092255
13: -9.2798910, 9.7542610, -9.2506313, 9.8155231, -16.3650589, 16.2862511
14: -86.1930695, -59.4875374, -86.0828857, -59.4894485, -19.9688034, 19.8255043
15: -29.5785465, -11.9120779, -29.5667572, -11.8959179, -12.1530724, 12.1346436
16: -43.3897896, -22.5518684, -43.3385239, -22.4853573, -16.2943115, 16.1996269
17: -100.0261917, -70.0151672, -99.9417572, -69.9545441, -22.1276245, 22.1136093
18: -17.7570305, 3.4519157, -17.8032169, 3.4439168, -13.6609116, 13.7147217
19: -21.0161896, -6.4487696, -20.9972572, -6.3957224, -12.4541245, 12.3942108
20: -8.1948929, 5.5805559, -8.2156010, 5.5739594, -13.7688522, 13.7961569
21: -30.4875851, -12.1480818, -30.4627762, -12.1046677, -16.1105652, 16.0834274
22: -24.7959805, -8.3610516, -24.7827415, -8.3482990, -12.1427383, 12.1578217
23: -16.8896904, 0.1482162, -16.8757896, 0.1443182, -14.1127396, 14.1414948
24: -8.0163097, 6.8941870, -8.0550079, 6.8801374, -12.7546349, 12.8221893
25: -4.5961103, 11.7290554, -4.6012187, 11.7260723, -14.1465912, 14.1634750
26: -23.0566998, -1.5704155, -23.1137466, -1.5849819, -18.2736893, 18.3783569
27: -17.8049316, -3.7967057, -17.8567657, -3.8144307, -12.8615799, 12.9392815
28: -3.3347309, 16.1642876, -3.3562617, 16.1451569, -15.9527512, 15.9784317
29: -41.7353516, -23.3522415, -41.7273254, -23.3131332, -14.5321198, 14.5551376
30: -11.7978029, 7.2474232, -11.8590984, 7.2184491, -17.6981964, 17.8057556
31: -22.8966427, -4.3982730, -22.8785477, -4.3484383, -15.3216782, 15.2146835
32: -3.7881198, 10.6062326, -3.7795873, 10.5966806, -11.2366295, 11.2192421
33: 10.5028458, 30.9258308, 10.5014620, 30.8863392, -16.2686234, 16.2978287
34: 11.2706947, 29.0128746, 11.1748495, 28.9460373, -11.3361092, 11.5453072
35: 22.9134636, 40.5302696, 22.8809338, 40.4719696, -11.2943611, 11.3506889
36: 17.9414711, 34.5525627, 17.8916054, 34.5112038, -12.3278961, 12.4295921
37: 7.8686867, 28.1307068, 7.8329630, 28.1198654, -16.7379227, 16.7640991
38: 6.5874844, 26.5927963, 6.5321527, 26.5598907, -14.3626328, 14.4597282
39: 5.7171001, 25.9480762, 5.7358041, 25.9513779, -16.2174072, 16.1967010
40: 0.5920010, 19.8888760, 0.5416799, 19.8559017, -12.5382690, 12.6505089
41: -4.0830507, 9.1085243, -4.0801368, 9.0892496, -10.9352341, 10.9766464
42: -27.5794868, -10.8368359, -27.5784740, -10.8434248, -11.5092697, 11.5287285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 755

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1872605, upper bound: 5.2121951
time: 6.99 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2033733, upper bound: 5.2121951
time: 7.12 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -57.6827240, -32.6131859, -57.6424942, -32.5474167, -17.5961227, 17.4893188
1: -39.2172318, -20.1973114, -39.1864395, -20.1433907, -11.9432945, 11.8526344
2: -27.2513695, -11.1546926, -27.2269306, -11.1061649, -11.0770340, 10.9876709
3: -31.5739975, -14.0833073, -31.5704060, -14.0534124, -10.9727249, 10.9413872
4: -29.4140396, -8.6281233, -29.3812580, -8.5543633, -14.2460556, 14.1527481
5: -31.7580070, -13.5358047, -31.7448235, -13.4851332, -12.2102051, 12.1518707
6: -14.8916121, 2.9160519, -14.8939972, 2.8808398, -11.5604477, 11.6305237
7: -46.6793060, -25.5320568, -46.6378250, -25.4850883, -12.0850105, 11.9764824
8: -41.4708023, -19.8298950, -41.4304733, -19.8097496, -10.7450447, 10.6708393
9: -24.2879219, -5.1259975, -24.2620735, -5.0512600, -16.5347366, 16.4301376
10: -52.1322021, -29.6381378, -52.0663147, -29.5171700, -17.2092438, 17.0586929
11: -47.9259796, -27.0997543, -47.9076004, -27.0839481, -15.0245209, 15.1218910
12: -13.3404608, 5.9193258, -13.3455296, 5.9341898, -15.4086456, 15.4233742
13: -9.2896643, 9.7548018, -9.2777967, 9.8607998, -16.4199219, 16.3035583
14: -86.1952591, -59.4897842, -86.0931396, -59.4930649, -19.9662094, 19.8661499
15: -29.5800056, -11.9116144, -29.5709114, -11.8809013, -12.1660767, 12.1354942
16: -43.4016228, -22.5527573, -43.3682213, -22.4365883, -16.3356857, 16.2186089
17: -100.0379181, -70.0158463, -99.9695435, -69.9353638, -22.1328735, 22.1588135
18: -17.7585754, 3.4539263, -17.8233910, 3.4516540, -13.6693878, 13.7623291
19: -21.0197601, -6.4488301, -21.0064812, -6.3693762, -12.4800034, 12.3983994
20: -8.1953526, 5.5820765, -8.2280149, 5.5798068, -13.7751598, 13.8100910
21: -30.4935570, -12.1480436, -30.4776134, -12.0840540, -16.1181335, 16.1066437
22: -24.7971745, -8.3610096, -24.7874355, -8.3321180, -12.1624680, 12.1661186
23: -16.8902950, 0.1494927, -16.8928566, 0.1495490, -14.1159592, 14.1666412
24: -8.0167255, 6.9030900, -8.0954390, 6.8993907, -12.7693405, 12.8666954
25: -4.5968523, 11.7299128, -4.6114254, 11.7294388, -14.1447449, 14.1895447
26: -23.0567894, -1.5705123, -23.1269302, -1.5797057, -18.2824326, 18.3975372
27: -17.8058662, -3.7869334, -17.9013748, -3.7929597, -12.8763275, 12.9860001
28: -3.3353426, 16.1669540, -3.3838372, 16.1529961, -15.9590530, 16.0042191
29: -41.7382889, -23.3520622, -41.7346497, -23.2972698, -14.5321121, 14.5752258
30: -11.7993383, 7.2611589, -11.9229708, 7.2502933, -17.7217712, 17.8736115
31: -22.8990154, -4.3984361, -22.8866806, -4.3215504, -15.3516235, 15.2209892
32: -3.7930775, 10.6087179, -3.7949586, 10.6257582, -11.2813644, 11.2329254
33: 10.5024405, 30.9256477, 10.4885674, 30.8905392, -16.2929611, 16.3049850
34: 11.2697172, 29.0264778, 11.1062498, 28.9763222, -11.3558311, 11.6381721
35: 22.9125710, 40.5366783, 22.8374062, 40.4858589, -11.3008919, 11.4014778
36: 17.9423714, 34.5580597, 17.8587875, 34.5227623, -12.3323975, 12.4633331
37: 7.8684993, 28.1316643, 7.8152127, 28.1226196, -16.7453918, 16.7772636
38: 6.5878525, 26.5968742, 6.4947042, 26.5742302, -14.3734283, 14.5107307
39: 5.7158661, 25.9485283, 5.7178221, 25.9841042, -16.2574234, 16.2136993
40: 0.5915709, 19.8919296, 0.5178113, 19.8632221, -12.5698013, 12.6551590
41: -4.0850444, 9.1104088, -4.0871763, 9.1005001, -10.9615936, 10.9801636
42: -27.5820770, -10.8356705, -27.5837440, -10.8326502, -11.5306587, 11.5279236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=112, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 755

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1960881, upper bound: 5.2121951
time: 5.49 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2121947, upper bound: 5.2121951
time: 29.13 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 36.73 seconds
IS_B2_A1_A2_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1951654, upper bound: 5.1914334
IS_B2_A1_A2_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.2112797, upper bound: 5.1914367
IS_B2_A2_A1_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1875283, upper bound: 5.1954958
IS_B2_A2_A1_A2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1875553, upper bound: 5.2116577
IS_B2_A2_A2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1642691, upper bound: 5.2113685
IS_B2_A2_A2_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1785456, upper bound: 5.2116833
IS_B2_A2_A2_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1712411, upper bound: 5.2113685
IS_B2_A2_A2_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1855060, upper bound: 5.2116833
IS_B2_A2_A2_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1700128, upper bound: 5.2113685
IS_B2_A2_A2_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1842833, upper bound: 5.2116833
IS_B2_A2_A2_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1788565, upper bound: 5.2113685
IS_B2_A2_A2_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1931200, upper bound: 5.2116833
IS_B2_A2_A2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1815266, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1976408, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1884794, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.2045883, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1872605, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.2033733, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.1960881, upper bound: 5.2121951
IS_B2_A2_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 36.73
Output dim: 35, lower bound: -5.2121947, upper bound: 5.2121951

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -57.6646690, -32.6454201, -57.6071587, -32.6303482, -17.5079880, 17.4382172
1: -39.2043839, -20.2197666, -39.1611900, -20.2094193, -11.8728104, 11.8213501
2: -27.2380314, -11.1736526, -27.2010040, -11.1663876, -11.0033684, 10.9569092
3: -31.5632744, -14.0908546, -31.5506840, -14.0881042, -10.9294205, 10.9171257
4: -29.3966084, -8.6532955, -29.3462811, -8.6432152, -14.1578827, 14.1105232
5: -31.7455997, -13.5520868, -31.7205925, -13.5441427, -12.1477089, 12.1140404
6: -14.8683777, 2.9161053, -14.8834572, 2.8771639, -11.5283394, 11.6146393
7: -46.6633492, -25.5510139, -46.6069527, -25.5418606, -12.0135918, 11.9435539
8: -41.4653625, -19.8592930, -41.4206505, -19.8459091, -10.6957703, 10.6396084
9: -24.2768154, -5.1459713, -24.2333794, -5.1385403, -16.4524689, 16.3926392
10: -52.1126785, -29.6712990, -52.0221252, -29.6633377, -17.0524673, 17.0039368
11: -47.9070396, -27.0951958, -47.8887520, -27.0950909, -14.9942856, 15.0440788
12: -13.3380547, 5.9054785, -13.3315258, 5.8962927, -15.3679886, 15.3934135
13: -9.2552147, 9.7292843, -9.2121897, 9.7390976, -16.2672958, 16.2398338
14: -86.1859665, -59.5347366, -86.0702362, -59.5094452, -19.9416962, 19.7507401
15: -29.5703030, -11.9331493, -29.5482597, -11.9249601, -12.1356926, 12.1021118
16: -43.3758850, -22.5722504, -43.3111000, -22.5632133, -16.2029572, 16.1481018
17: -99.9971313, -70.0436249, -99.8845596, -70.0284729, -22.0768433, 22.0351944
18: -17.7510986, 3.4309916, -17.7459774, 3.4099977, -13.6333389, 13.6481819
19: -20.9974594, -6.4497099, -20.9727707, -6.4449964, -12.3669434, 12.3762016
20: -8.1782093, 5.5761847, -8.1784782, 5.5653276, -13.7435369, 13.7546635
21: -30.4666691, -12.1491175, -30.4337749, -12.1462708, -16.0338593, 16.0626068
22: -24.7876110, -8.3615017, -24.7707043, -8.3608837, -12.1136780, 12.1357155
23: -16.8784828, 0.1487734, -16.8678036, 0.1457182, -14.0949707, 14.1260376
24: -8.0008745, 6.8827333, -8.0042858, 6.8583031, -12.7217216, 12.7636566
25: -4.5776262, 11.7255220, -4.5769672, 11.7189178, -14.1315155, 14.1545258
26: -23.0387993, -1.5800848, -23.0372429, -1.6012182, -18.2290878, 18.2807617
27: -17.7878399, -3.8108428, -17.7918930, -3.8409224, -12.8257980, 12.8667450
28: -3.3134468, 16.1606655, -3.3159015, 16.1394119, -15.9325867, 15.9520187
29: -41.7130432, -23.3535118, -41.6944199, -23.3549576, -14.4966888, 14.5188255
30: -11.7744808, 7.2279758, -11.7772751, 7.1828566, -17.6473312, 17.7099686
31: -22.8764076, -4.3995843, -22.8607960, -4.3938408, -15.2237167, 15.1983376
32: -3.7784684, 10.6011524, -3.7777750, 10.5862808, -11.1749039, 11.2074661
33: 10.5391855, 30.9241734, 10.5266113, 30.8827095, -16.2170868, 16.2766800
34: 11.2935772, 28.9887581, 11.2828970, 28.8973045, -11.2847939, 11.3965797
35: 22.9424744, 40.5213242, 22.9329224, 40.4555779, -11.2582436, 11.2932434
36: 17.9697266, 34.5427017, 17.9577503, 34.4913826, -12.2894707, 12.3591042
37: 7.8868823, 28.1203995, 7.8830438, 28.1011391, -16.7141876, 16.7120399
38: 6.6087523, 26.5811691, 6.5964279, 26.5368500, -14.3226700, 14.3690109
39: 5.7353382, 25.9452591, 5.7391515, 25.9402676, -16.1801758, 16.1858597
40: 0.6080141, 19.8720589, 0.6000304, 19.8243065, -12.5071564, 12.5852013
41: -4.0689383, 9.1039829, -4.0722156, 9.0786362, -10.9182243, 10.9708176
42: -27.5588417, -10.8422604, -27.5621014, -10.8603306, -11.4776993, 11.5153160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1669318, upper bound: 5.2115728
time: 5.82 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1811695, upper bound: 5.2118326
time: 5.47 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -57.7356529, -32.6154442, -57.6067162, -32.6197052, -17.5903168, 17.4629555
1: -39.2539978, -20.1976910, -39.1613235, -20.2018185, -11.9297104, 11.8392715
2: -27.2802849, -11.1540527, -27.2006836, -11.1600990, -11.0518379, 10.9722939
3: -31.5756130, -14.0784273, -31.5508461, -14.0845814, -10.9438400, 10.9307823
4: -29.4577274, -8.6284122, -29.3463287, -8.6349335, -14.2272797, 14.1312256
5: -31.7658768, -13.5355530, -31.7207336, -13.5388193, -12.1741257, 12.1307182
6: -14.8924885, 2.9735708, -14.8914213, 2.8759556, -11.5407715, 11.6801796
7: -46.6998749, -25.5318413, -46.6069870, -25.5351677, -12.0576477, 11.9603767
8: -41.5206833, -19.8303604, -41.4207687, -19.8358498, -10.7611275, 10.6627998
9: -24.3218002, -5.1244259, -24.2332764, -5.1319742, -16.5038834, 16.4116211
10: -52.1675606, -29.6341190, -52.0220871, -29.6529446, -17.1233292, 17.0345154
11: -47.9196243, -27.0719624, -47.8904648, -27.0959187, -14.9985962, 15.0961647
12: -13.3471327, 5.9244151, -13.3328066, 5.8965521, -15.3858490, 15.4117050
13: -9.3084259, 9.7546530, -9.2147179, 9.7471552, -16.3285522, 16.2659912
14: -86.3108292, -59.4890823, -86.0695038, -59.4923096, -20.0832901, 19.7853851
15: -29.6165676, -11.9120770, -29.5486088, -11.9181461, -12.1888123, 12.1195869
16: -43.4120407, -22.5506763, -43.3115845, -22.5564938, -16.2506180, 16.1683769
17: -100.0584183, -70.0153961, -99.8849030, -70.0185547, -22.1033173, 22.0524368
18: -17.7619781, 3.4398460, -17.7469635, 3.4105196, -13.6387253, 13.6608429
19: -21.0120296, -6.4375124, -20.9762020, -6.4448404, -12.3760452, 12.4049911
20: -8.1979418, 5.5988345, -8.1832476, 5.5650501, -13.7629919, 13.7820816
21: -30.4844589, -12.1318293, -30.4363918, -12.1467686, -16.0428009, 16.1084747
22: -24.8039398, -8.3471107, -24.7739258, -8.3608341, -12.1247063, 12.1689491
23: -16.8922119, 0.1667081, -16.8708286, 0.1461145, -14.0997772, 14.1666107
24: -8.0169983, 6.9086099, -8.0088730, 6.8584776, -12.7325058, 12.8060341
25: -4.5972295, 11.7548313, -4.5825419, 11.7190638, -14.1498718, 14.1913795
26: -23.0596733, -1.5642705, -23.0421753, -1.6006649, -18.2374649, 18.3377762
27: -17.8062477, -3.7825904, -17.7970486, -3.8408461, -12.8403854, 12.9061928
28: -3.3371084, 16.1948509, -3.3223867, 16.1398506, -15.9564667, 15.9920883
29: -41.7272873, -23.3327217, -41.6968727, -23.3551712, -14.5095215, 14.5516243
30: -11.7992172, 7.2845249, -11.7844305, 7.1828570, -17.6694565, 17.7758865
31: -22.8956261, -4.3817024, -22.8659706, -4.3936701, -15.2426071, 15.2230492
32: -3.7937186, 10.6212540, -3.7818706, 10.5857477, -11.1847572, 11.2299690
33: 10.4990139, 30.9759293, 10.5145817, 30.8827782, -16.2530746, 16.3398209
34: 11.2695293, 29.0338993, 11.2756205, 28.8973808, -11.3050308, 11.4488068
35: 22.9110260, 40.5544319, 22.9236469, 40.4553528, -11.2835464, 11.3321457
36: 17.9396172, 34.5698738, 17.9482155, 34.4912758, -12.3146667, 12.3963852
37: 7.8635521, 28.1332893, 7.8772407, 28.1008873, -16.7391586, 16.7300339
38: 6.5839348, 26.5885887, 6.5893993, 26.5357780, -14.3504944, 14.3877678
39: 5.7043076, 25.9471741, 5.7309012, 25.9372444, -16.2127304, 16.2017365
40: 0.5872173, 19.9021721, 0.5944166, 19.8244801, -12.5266342, 12.6180191
41: -4.0843377, 9.1393127, -4.0768089, 9.0780048, -10.9310493, 11.0100975
42: -27.5765247, -10.7952442, -27.5676231, -10.8610697, -11.4904938, 11.5689697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 739

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1819839, upper bound: 5.2116675
time: 17.09 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1971599, upper bound: 5.2117176
time: 5.76 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -57.6690025, -32.6454239, -57.6184044, -32.6124153, -17.5211792, 17.4449234
1: -39.2084961, -20.2200871, -39.1704597, -20.1927109, -11.8893929, 11.8250771
2: -27.2392006, -11.1735077, -27.2045212, -11.1580143, -11.0131683, 10.9591141
3: -31.5637817, -14.0943756, -31.5520134, -14.0895424, -10.9330025, 10.9236984
4: -29.4006767, -8.6530685, -29.3558102, -8.6128960, -14.1895065, 14.1177979
5: -31.7457752, -13.5519648, -31.7222099, -13.5367556, -12.1484795, 12.1295052
6: -14.8682575, 2.9155045, -14.8831720, 2.8773708, -11.5297928, 11.6160736
7: -46.6659355, -25.5518265, -46.6132660, -25.5382690, -12.0181274, 11.9462776
8: -41.4653549, -19.8595181, -41.4209442, -19.8429775, -10.7078094, 10.6363716
9: -24.2842484, -5.1459150, -24.2519608, -5.0922441, -16.5008163, 16.4054184
10: -52.1224594, -29.6705666, -52.0433502, -29.5999775, -17.1204605, 17.0168381
11: -47.9082832, -27.1024570, -47.8864365, -27.1051426, -14.9916840, 15.0749588
12: -13.3404417, 5.9071598, -13.3432827, 5.9225688, -15.3966103, 15.4062347
13: -9.2640371, 9.7296991, -9.2357864, 9.7823887, -16.3164825, 16.2538910
14: -86.1876831, -59.5369797, -86.0786743, -59.5131073, -19.9353867, 19.7880478
15: -29.5712242, -11.9327812, -29.5512810, -11.9106331, -12.1459770, 12.1021996
16: -43.3861313, -22.5732002, -43.3366699, -22.5170345, -16.2408676, 16.1642761
17: -100.0082626, -70.0444489, -99.9109650, -70.0097198, -22.0765533, 22.0772629
18: -17.7523251, 3.4329515, -17.7652626, 3.4171247, -13.6393318, 13.6916580
19: -21.0004005, -6.4497619, -20.9799614, -6.4211884, -12.3913651, 12.3794975
20: -8.1785583, 5.5771894, -8.1903152, 5.5692310, -13.7477894, 13.7675047
21: -30.4724789, -12.1491585, -30.4468918, -12.1268902, -16.0401764, 16.0838699
22: -24.7887726, -8.3615017, -24.7749214, -8.3447800, -12.1287231, 12.1428795
23: -16.8789978, 0.1499832, -16.8841381, 0.1504054, -14.0976791, 14.1491623
24: -8.0012093, 6.8915796, -8.0429916, 6.8772898, -12.7344475, 12.8042755
25: -4.5781984, 11.7262888, -4.5863018, 11.7220688, -14.1290131, 14.1782455
26: -23.0386753, -1.5811400, -23.0499268, -1.5995774, -18.2358170, 18.2993774
27: -17.7886086, -3.8024213, -17.8333187, -3.8222547, -12.8387985, 12.9081841
28: -3.3139391, 16.1624985, -3.3422434, 16.1446953, -15.9379425, 15.9756546
29: -41.7158966, -23.3533745, -41.7009468, -23.3395081, -14.4890747, 14.5367279
30: -11.7758007, 7.2400556, -11.8381519, 7.2102766, -17.6660614, 17.7700882
31: -22.8787422, -4.3998284, -22.8670101, -4.3693585, -15.2534790, 15.2033691
32: -3.7829621, 10.6033087, -3.7909470, 10.6142607, -11.2184639, 11.2189026
33: 10.5390644, 30.9239635, 10.5161705, 30.8868637, -16.2406235, 16.2814865
34: 11.2927494, 29.0013294, 11.2201357, 28.9229336, -11.3017082, 11.4863091
35: 22.9418259, 40.5270004, 22.8933945, 40.4676666, -11.2642212, 11.3385010
36: 17.9706974, 34.5479393, 17.9284592, 34.5017433, -12.2929611, 12.3877831
37: 7.8869729, 28.1213684, 7.8671784, 28.1039028, -16.7215195, 16.7206612
38: 6.6092181, 26.5835400, 6.5640979, 26.5474701, -14.3317490, 14.4116211
39: 5.7343497, 25.9456940, 5.7245417, 25.9728222, -16.2175522, 16.1995773
40: 0.6077518, 19.8750706, 0.5769300, 19.8309517, -12.5367355, 12.5822601
41: -4.0708714, 9.1056070, -4.0782814, 9.0892763, -10.9417381, 10.9720039
42: -27.5614548, -10.8413095, -27.5673656, -10.8503275, -11.4979515, 11.5139809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1739048, upper bound: 5.2115728
time: 11.51 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1881229, upper bound: 5.2118326
time: 9.96 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -57.7399445, -32.6154366, -57.6179771, -32.6018105, -17.6034927, 17.4696617
1: -39.2581520, -20.1980247, -39.1705627, -20.1851120, -11.9462929, 11.8429909
2: -27.2814560, -11.1538544, -27.2042122, -11.1516876, -11.0616493, 10.9744835
3: -31.5761452, -14.0819454, -31.5521851, -14.0860338, -10.9474525, 10.9373627
4: -29.4618187, -8.6281796, -29.3558483, -8.6046190, -14.2589188, 14.1384888
5: -31.7660332, -13.5354748, -31.7223263, -13.5314026, -12.1748505, 12.1461983
6: -14.8923683, 2.9730377, -14.8911190, 2.8761415, -11.5422401, 11.6816025
7: -46.7025223, -25.5326385, -46.6132851, -25.5315857, -12.0621529, 11.9631119
8: -41.5206985, -19.8305340, -41.4210167, -19.8329811, -10.7731476, 10.6595783
9: -24.3292122, -5.1243544, -24.2518482, -5.0856934, -16.5522614, 16.4243698
10: -52.1773643, -29.6334305, -52.0432892, -29.5896111, -17.1913300, 17.0474091
11: -47.9208412, -27.0791779, -47.8881493, -27.1059837, -14.9960022, 15.1270142
12: -13.3495064, 5.9260774, -13.3445740, 5.9228082, -15.4144745, 15.4244957
13: -9.3172541, 9.7550716, -9.2383423, 9.7904453, -16.3777542, 16.2800217
14: -86.3125763, -59.4913635, -86.0779800, -59.4959488, -20.0769806, 19.8226929
15: -29.6175251, -11.9116707, -29.5516357, -11.9038162, -12.1990738, 12.1196747
16: -43.4222336, -22.5516071, -43.3371429, -22.5103149, -16.2885284, 16.1845512
17: -100.0695953, -70.0161743, -99.9113464, -69.9997559, -22.1030426, 22.0944977
18: -17.7632675, 3.4417844, -17.7662621, 3.4176662, -13.6447029, 13.7043800
19: -21.0149765, -6.4375868, -20.9834213, -6.4210339, -12.4004669, 12.4082756
20: -8.1982718, 5.5998230, -8.1950903, 5.5689712, -13.7672424, 13.7949133
21: -30.4902267, -12.1318884, -30.4494858, -12.1274109, -16.0491333, 16.1297836
22: -24.8050365, -8.3470984, -24.7781620, -8.3447847, -12.1397438, 12.1761169
23: -16.8927574, 0.1678773, -16.8871384, 0.1507981, -14.1024628, 14.1897202
24: -8.0173178, 6.9174623, -8.0475855, 6.8774729, -12.7452164, 12.8466873
25: -4.5978198, 11.7556152, -4.5919085, 11.7221994, -14.1473465, 14.2150955
26: -23.0595455, -1.5653791, -23.0548611, -1.5989628, -18.2442169, 18.3563995
27: -17.8069649, -3.7741425, -17.8384781, -3.8221893, -12.8533936, 12.9476280
28: -3.3376331, 16.1966801, -3.3486722, 16.1451550, -15.9617996, 16.0157318
29: -41.7301178, -23.3325996, -41.7033997, -23.3397007, -14.5018997, 14.5695419
30: -11.8005543, 7.2966223, -11.8453236, 7.2103014, -17.6881638, 17.8360443
31: -22.8980045, -4.3819485, -22.8721619, -4.3691659, -15.2723312, 15.2280998
32: -3.7982080, 10.6233807, -3.7950470, 10.6137295, -11.2283401, 11.2414207
33: 10.4989023, 30.9757538, 10.5040970, 30.8869209, -16.2765961, 16.3446808
34: 11.2686996, 29.0465279, 11.2128725, 28.9230347, -11.3219414, 11.5385361
35: 22.9103508, 40.5601006, 22.8840866, 40.4674301, -11.2895279, 11.3774338
36: 17.9405880, 34.5751305, 17.9189110, 34.5016365, -12.3181610, 12.4250755
37: 7.8636417, 28.1341953, 7.8613801, 28.1036606, -16.7465363, 16.7386551
38: 6.5844107, 26.5909691, 6.5570440, 26.5464230, -14.3595886, 14.4304161
39: 5.7033277, 25.9475517, 5.7162824, 25.9697762, -16.2500916, 16.2154236
40: 0.5869212, 19.9051037, 0.5712919, 19.8311119, -12.5561867, 12.6150856
41: -4.0862770, 9.1409187, -4.0828705, 9.0886660, -10.9545555, 11.0112991
42: -27.5791302, -10.7942371, -27.5728836, -10.8510218, -11.5107384, 11.5676498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1900472, upper bound: 5.2115831
time: 8.79 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2042327, upper bound: 5.2118376
time: 5.57 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -57.6752319, -32.6441498, -57.6276474, -32.5787659, -17.5655785, 17.4480171
1: -39.2118607, -20.2186947, -39.1759415, -20.1695766, -11.9140358, 11.8249702
2: -27.2492542, -11.1729050, -27.2225723, -11.1227093, -11.0574188, 10.9662971
3: -31.5724678, -14.0906391, -31.5684261, -14.0576477, -10.9627724, 10.9232788
4: -29.4075680, -8.6522398, -29.3684635, -8.5977325, -14.1971588, 14.1195793
5: -31.7570667, -13.5516624, -31.7428322, -13.5000143, -12.1996841, 12.1205025
6: -14.8684053, 2.9145756, -14.8835096, 2.8794541, -11.5342369, 11.6165962
7: -46.6757736, -25.5505676, -46.6309624, -25.4971161, -12.0706596, 11.9528389
8: -41.4696579, -19.8586922, -41.4296455, -19.8268776, -10.7170067, 10.6443634
9: -24.2785797, -5.1453323, -24.2398605, -5.1091022, -16.4709930, 16.3949280
10: -52.1218643, -29.6692028, -52.0428085, -29.5991879, -17.1172180, 17.0139542
11: -47.9145126, -27.0949631, -47.9046860, -27.0752678, -15.0066833, 15.0766602
12: -13.3324003, 5.9089241, -13.3294687, 5.9021993, -15.3664932, 15.3951378
13: -9.2699757, 9.7307224, -9.2463360, 9.8053684, -16.3447037, 16.2580299
14: -86.1892090, -59.5361404, -86.0811920, -59.5103874, -19.9442444, 19.7746620
15: -29.5747528, -11.9320450, -29.5651016, -11.9045200, -12.1403542, 12.1125069
16: -43.3872757, -22.5714531, -43.3374557, -22.4937973, -16.2831726, 16.1782379
17: -100.0224457, -70.0435867, -99.9401398, -69.9667892, -22.1155701, 22.0975418
18: -17.7532806, 3.4469051, -17.8016033, 3.4416974, -13.6527481, 13.7080956
19: -21.0060177, -6.4496884, -20.9927254, -6.3961067, -12.4393463, 12.3870850
20: -8.1805429, 5.5792656, -8.2094078, 5.5733948, -13.7539377, 13.7886734
21: -30.4773159, -12.1489878, -30.4583569, -12.1050577, -16.0917282, 16.0740623
22: -24.7860336, -8.3614845, -24.7783985, -8.3484688, -12.1274567, 12.1505508
23: -16.8802738, 0.1464832, -16.8716621, 0.1435884, -14.0955963, 14.1323967
24: -8.0021152, 6.8928967, -8.0488720, 6.8795996, -12.7359314, 12.8128204
25: -4.5796824, 11.7283173, -4.5940847, 11.7257462, -14.1290741, 14.1551361
26: -23.0419769, -1.5738568, -23.1073742, -1.5864902, -18.2468185, 18.3637695
27: -17.7895718, -3.7986827, -17.8501301, -3.8152957, -12.8434372, 12.9297867
28: -3.3156052, 16.1623516, -3.3479970, 16.1443176, -15.9328308, 15.9681702
29: -41.7251472, -23.3536034, -41.7229156, -23.3137016, -14.5183029, 14.5481911
30: -11.7770214, 7.2447028, -11.8500748, 7.2172461, -17.6750412, 17.7934113
31: -22.8809853, -4.3993692, -22.8717499, -4.3489089, -15.3044891, 15.2061119
32: -3.7748041, 10.6045618, -3.7738433, 10.5959549, -11.2228355, 11.2118034
33: 10.5376673, 30.9244499, 10.5165024, 30.8857231, -16.2348099, 16.2819977
34: 11.2920189, 29.0109711, 11.1840229, 28.9451752, -11.3135948, 11.5340500
35: 22.9409847, 40.5298386, 22.8928356, 40.4717712, -11.2679825, 11.3388786
36: 17.9692039, 34.5519409, 17.9035568, 34.5109596, -12.3008919, 12.4174118
37: 7.8862653, 28.1298752, 7.8405609, 28.1194782, -16.7222824, 16.7560043
38: 6.6087418, 26.5890770, 6.5413342, 26.5582848, -14.3365326, 14.4436531
39: 5.7421560, 25.9463787, 5.7466307, 25.9506340, -16.1929703, 16.1830673
40: 0.6081271, 19.8868046, 0.5486670, 19.8549652, -12.5207748, 12.6397858
41: -4.0691161, 9.1068611, -4.0741324, 9.0885296, -10.9200859, 10.9686241
42: -27.5629921, -10.8399897, -27.5713711, -10.8447762, -11.4914207, 11.5188637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1726750, upper bound: 5.2115728
time: 5.88 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1869036, upper bound: 5.2118326
time: 10.12 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -57.7461929, -32.6141739, -57.6271591, -32.5681572, -17.6478958, 17.4727821
1: -39.2614975, -20.1965981, -39.1760406, -20.1619644, -11.9709396, 11.8429031
2: -27.2915421, -11.1532717, -27.2222404, -11.1163807, -11.1059227, 10.9816551
3: -31.5848160, -14.0782299, -31.5685940, -14.0541372, -10.9771843, 10.9369507
4: -29.4686813, -8.6273546, -29.3685074, -8.5893736, -14.2665558, 14.1402702
5: -31.7773476, -13.5351200, -31.7429237, -13.4946995, -12.2260780, 12.1371689
6: -14.8925209, 2.9720802, -14.8914347, 2.8782415, -11.5466728, 11.6821327
7: -46.7123909, -25.5313511, -46.6310310, -25.4903908, -12.1147156, 11.9696732
8: -41.5249786, -19.8297100, -41.4297333, -19.8168392, -10.7823715, 10.6675606
9: -24.3235970, -5.1237841, -24.2397537, -5.1025581, -16.5223923, 16.4138947
10: -52.1767769, -29.6320496, -52.0427475, -29.5887947, -17.1881104, 17.0445328
11: -47.9271011, -27.0717163, -47.9064407, -27.0761032, -15.0110092, 15.1287270
12: -13.3414860, 5.9278331, -13.3307209, 5.9024463, -15.3843918, 15.4134216
13: -9.3232298, 9.7560654, -9.2489014, 9.8134117, -16.4059677, 16.2841187
14: -86.3141098, -59.4905052, -86.0805206, -59.4932480, -20.0858536, 19.8093185
15: -29.6210213, -11.9109592, -29.5654545, -11.8976984, -12.1934509, 12.1299858
16: -43.4234161, -22.5498619, -43.3378868, -22.4870491, -16.3308563, 16.1985168
17: -100.0837631, -70.0153046, -99.9405289, -69.9569168, -22.1420898, 22.1147461
18: -17.7642059, 3.4557595, -17.8025646, 3.4422579, -13.6581535, 13.7207985
19: -21.0205994, -6.4374804, -20.9961967, -6.3959308, -12.4484329, 12.4158783
20: -8.2002478, 5.6019087, -8.2141829, 5.5731163, -13.7733641, 13.8160915
21: -30.4951172, -12.1316996, -30.4609356, -12.1055489, -16.1006622, 16.1199570
22: -24.8022652, -8.3471088, -24.7815819, -8.3484726, -12.1384621, 12.1837883
23: -16.8940296, 0.1644187, -16.8746738, 0.1439826, -14.1003647, 14.1729240
24: -8.0182390, 6.9187646, -8.0534773, 6.8797660, -12.7466812, 12.8552513
25: -4.5992584, 11.7576609, -4.5997014, 11.7259140, -14.1473923, 14.1919861
26: -23.0628471, -1.5580096, -23.1123085, -1.5859118, -18.2551880, 18.4208069
27: -17.8079510, -3.7704163, -17.8552933, -3.8151975, -12.8579865, 12.9692383
28: -3.3392909, 16.1965179, -3.3544955, 16.1447678, -15.9567032, 16.0082397
29: -41.7393799, -23.3328094, -41.7253876, -23.3139381, -14.5311279, 14.5809593
30: -11.8017750, 7.3012342, -11.8572493, 7.2172928, -17.6971512, 17.8593826
31: -22.9001884, -4.3814287, -22.8768940, -4.3487234, -15.3234024, 15.2308502
32: -3.7900522, 10.6246605, -3.7779465, 10.5954332, -11.2327118, 11.2343445
33: 10.4974785, 30.9762497, 10.5044460, 30.8857765, -16.2708054, 16.3451614
34: 11.2679787, 29.0561447, 11.1767302, 28.9452972, -11.3338432, 11.5863075
35: 22.9095707, 40.5629463, 22.8835545, 40.4715424, -11.2932816, 11.3778076
36: 17.9390926, 34.5790939, 17.8939991, 34.5108337, -12.3260803, 12.4546738
37: 7.8629246, 28.1427460, 7.8347397, 28.1192188, -16.7472916, 16.7740021
38: 6.5839171, 26.5964890, 6.5343151, 26.5571938, -14.3643265, 14.4624138
39: 5.7111058, 25.9482689, 5.7383609, 25.9475594, -16.2255020, 16.1989441
40: 0.5872822, 19.9169006, 0.5429983, 19.8551292, -12.5402374, 12.6726303
41: -4.0845332, 9.1421919, -4.0787044, 9.0879288, -10.9329147, 11.0079002
42: -27.5806389, -10.7929306, -27.5769043, -10.8454971, -11.5042076, 11.5725250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1888206, upper bound: 5.2115831
time: 15.48 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2030181, upper bound: 5.2118376
time: 5.55 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -57.6797180, -32.6440773, -57.6411705, -32.5607300, -17.5799561, 17.4573364
1: -39.2160835, -20.2189865, -39.1859283, -20.1527214, -11.9325905, 11.8297386
2: -27.2504578, -11.1726837, -27.2265358, -11.1139078, -11.0682564, 10.9691200
3: -31.5730419, -14.0940943, -31.5699883, -14.0580540, -10.9664421, 10.9302864
4: -29.4121494, -8.6519375, -29.3804474, -8.5646420, -14.2340546, 14.1279869
5: -31.7572346, -13.5514879, -31.7445068, -13.4918909, -12.2027473, 12.1361465
6: -14.8682919, 2.9141550, -14.8839607, 2.8800135, -11.5361824, 11.6189651
7: -46.6784859, -25.5513515, -46.6374817, -25.4934006, -12.0753555, 11.9557686
8: -41.4696503, -19.8588867, -41.4299698, -19.8222656, -10.7312012, 10.6412354
9: -24.2864761, -5.1451888, -24.2614594, -5.0595083, -16.5247421, 16.4097977
10: -52.1316948, -29.6683388, -52.0660667, -29.5302448, -17.1952820, 17.0276871
11: -47.9159164, -27.1021080, -47.9032288, -27.0849781, -15.0056152, 15.1116943
12: -13.3349228, 5.9109068, -13.3431721, 5.9305925, -15.3977585, 15.4092903
13: -9.2797318, 9.7312469, -9.2735424, 9.8506956, -16.3995209, 16.2752914
14: -86.1914062, -59.5383759, -86.0914841, -59.5140152, -19.9416504, 19.8153229
15: -29.5762253, -11.9315844, -29.5692711, -11.8895130, -12.1533813, 12.1133690
16: -43.3991127, -22.5723476, -43.3671494, -22.4450455, -16.3245239, 16.1972084
17: -100.0341568, -70.0443573, -99.9678574, -69.9476700, -22.1207886, 22.1427231
18: -17.7547779, 3.4489028, -17.8218117, 3.4494781, -13.6612549, 13.7556915
19: -21.0096016, -6.4497337, -21.0019913, -6.3697643, -12.4652100, 12.3912735
20: -8.1809998, 5.5808039, -8.2218189, 5.5792413, -13.7602406, 13.8026228
21: -30.4833126, -12.1489830, -30.4731827, -12.0844603, -16.0992889, 16.0972900
22: -24.7872334, -8.3614502, -24.7831078, -8.3322773, -12.1471863, 12.1588211
23: -16.8809261, 0.1477587, -16.8887539, 0.1488196, -14.0987396, 14.1575241
24: -8.0025368, 6.9018211, -8.0893211, 6.8988371, -12.7505951, 12.8573456
25: -4.5803781, 11.7291737, -4.6043119, 11.7291508, -14.1272354, 14.1811905
26: -23.0420513, -1.5739555, -23.1205463, -1.5811651, -18.2554779, 18.3829041
27: -17.7904587, -3.7889276, -17.8947411, -3.7937956, -12.8581467, 12.9764671
28: -3.3162336, 16.1650600, -3.3755624, 16.1521473, -15.9391708, 15.9939728
29: -41.7280350, -23.3533936, -41.7302437, -23.2978897, -14.5182724, 14.5682449
30: -11.7785997, 7.2584076, -11.9139643, 7.2491274, -17.6986160, 17.8613052
31: -22.8833351, -4.3995495, -22.8798523, -4.3220367, -15.3344269, 15.2124138
32: -3.7797427, 10.6070232, -3.7892385, 10.6250134, -11.2675858, 11.2254982
33: 10.5372610, 30.9243050, 10.5035620, 30.8899460, -16.2591400, 16.2891769
34: 11.2910175, 29.0245590, 11.1154861, 28.9754906, -11.3333054, 11.6269455
35: 22.9401093, 40.5362396, 22.8492908, 40.4856491, -11.2745132, 11.3896561
36: 17.9700985, 34.5574265, 17.8707542, 34.5224915, -12.3053856, 12.4511604
37: 7.8860779, 28.1308289, 7.8227835, 28.1222382, -16.7297363, 16.7692032
38: 6.6090832, 26.5931206, 6.5038738, 26.5725822, -14.3473053, 14.4946442
39: 5.7409644, 25.9468803, 5.7286215, 25.9833488, -16.2329941, 16.2001190
40: 0.6076889, 19.8898392, 0.5247498, 19.8623314, -12.5523071, 12.6444283
41: -4.0711441, 9.1087456, -4.0811930, 9.0997448, -10.9464684, 10.9721413
42: -27.5655956, -10.8387928, -27.5766239, -10.8340263, -11.5128021, 11.5180817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1815210, upper bound: 5.2115728
time: 12.38 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1957330, upper bound: 5.2118325
time: 14.74 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -57.7506485, -32.6141548, -57.6407433, -32.5501175, -17.6622810, 17.4820595
1: -39.2657013, -20.1968918, -39.1860619, -20.1451550, -11.9895096, 11.8476944
2: -27.2927322, -11.1530323, -27.2262325, -11.1076412, -11.1167564, 10.9844971
3: -31.5854130, -14.0816345, -31.5701790, -14.0545464, -10.9809113, 10.9439583
4: -29.4732933, -8.6270399, -29.3804665, -8.5563183, -14.3034630, 14.1486702
5: -31.7775192, -13.5349312, -31.7445831, -13.4865494, -12.2291527, 12.1528358
6: -14.8924341, 2.9716377, -14.8918915, 2.8787928, -11.5486145, 11.6844902
7: -46.7150307, -25.5321465, -46.6375122, -25.4867134, -12.1193848, 11.9725914
8: -41.5249901, -19.8298836, -41.4300575, -19.8122215, -10.7965584, 10.6644363
9: -24.3314629, -5.1236382, -24.2613487, -5.0530014, -16.5761261, 16.4287720
10: -52.1865997, -29.6311951, -52.0659752, -29.5198574, -17.2661667, 17.0581970
11: -47.9284630, -27.0788803, -47.9049606, -27.0857830, -15.0099487, 15.1637878
12: -13.3439713, 5.9298010, -13.3444767, 5.9308209, -15.4156342, 15.4275627
13: -9.3329391, 9.7566032, -9.2760496, 9.8587198, -16.4607849, 16.3014145
14: -86.3162918, -59.4927559, -86.0907288, -59.4968796, -20.0832748, 19.8499985
15: -29.6225033, -11.9105072, -29.5696087, -11.8826942, -12.2064743, 12.1308327
16: -43.4352455, -22.5507736, -43.3675957, -22.4382801, -16.3722534, 16.2174568
17: -100.0954666, -70.0160980, -99.9682236, -69.9377136, -22.1473236, 22.1599884
18: -17.7657433, 3.4577665, -17.8227768, 3.4500008, -13.6666603, 13.7684135
19: -21.0241699, -6.4375424, -21.0054092, -6.3695755, -12.4742851, 12.4200897
20: -8.2007065, 5.6034245, -8.2265930, 5.5789571, -13.7796631, 13.8300171
21: -30.5011024, -12.1316710, -30.4757900, -12.0849352, -16.1082306, 16.1431885
22: -24.8035088, -8.3470240, -24.7863121, -8.3322773, -12.1581841, 12.1920815
23: -16.8946667, 0.1656830, -16.8917656, 0.1492167, -14.1035614, 14.1980743
24: -8.0186663, 6.9276867, -8.0939350, 6.8990049, -12.7613869, 12.8997383
25: -4.5999508, 11.7584953, -4.6099005, 11.7292747, -14.1455460, 14.2180710
26: -23.0629158, -1.5581696, -23.1254807, -1.5805891, -18.2638855, 18.4399567
27: -17.8088455, -3.7606201, -17.8998928, -3.7937322, -12.8727570, 13.0159073
28: -3.3399153, 16.1992378, -3.3820469, 16.1525803, -15.9630585, 16.0340271
29: -41.7423248, -23.3326378, -41.7327003, -23.2980804, -14.5310822, 14.6010437
30: -11.8033018, 7.3149652, -11.9211130, 7.2491446, -17.7207108, 17.9272461
31: -22.9025536, -4.3816295, -22.8850651, -4.3218503, -15.3532944, 15.2371559
32: -3.7950187, 10.6271477, -3.7933230, 10.6244888, -11.2774467, 11.2480774
33: 10.4970894, 30.9760952, 10.4915791, 30.8899918, -16.2950974, 16.3523407
34: 11.2670288, 29.0697231, 11.1081867, 28.9755974, -11.3535576, 11.6791725
35: 22.9086533, 40.5693474, 22.8399696, 40.4854279, -11.2998161, 11.4285812
36: 17.9400005, 34.5845947, 17.8612175, 34.5223961, -12.3305817, 12.4884338
37: 7.8627381, 28.1436882, 7.8169537, 28.1219730, -16.7547455, 16.7871971
38: 6.5842657, 26.6005630, 6.4968462, 26.5715313, -14.3751373, 14.5134087
39: 5.7099028, 25.9487228, 5.7204165, 25.9802742, -16.2655411, 16.2159729
40: 0.5868940, 19.9199181, 0.5191402, 19.8624687, -12.5717583, 12.6772690
41: -4.0865259, 9.1440439, -4.0857525, 9.0991373, -10.9592667, 11.0114517
42: -27.5832615, -10.7917233, -27.5821552, -10.8347702, -11.5255890, 11.5717278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=111, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 722

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1976617, upper bound: 5.2115831
time: 5.78 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2118370, upper bound: 5.2118376
time: 5.36 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 13.23 seconds
IS_B2_A2_A2_A2_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1669318, upper bound: 5.2115728
IS_B2_A2_A2_A2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1811695, upper bound: 5.2118326
IS_B2_A2_A2_A2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1819839, upper bound: 5.2116675
IS_B2_A2_A2_A2_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1971599, upper bound: 5.2117176
IS_B2_A2_A2_A2_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1739048, upper bound: 5.2115728
IS_B2_A2_A2_A2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1881229, upper bound: 5.2118326
IS_B2_A2_A2_A2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1900472, upper bound: 5.2115831
IS_B2_A2_A2_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.2042327, upper bound: 5.2118376
IS_B2_A2_A2_A2_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1726750, upper bound: 5.2115728
IS_B2_A2_A2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1869036, upper bound: 5.2118326
IS_B2_A2_A2_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1888206, upper bound: 5.2115831
IS_B2_A2_A2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.2030181, upper bound: 5.2118376
IS_B2_A2_A2_A2_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1815210, upper bound: 5.2115728
IS_B2_A2_A2_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1957330, upper bound: 5.2118325
IS_B2_A2_A2_A2_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.1976617, upper bound: 5.2115831
IS_B2_A2_A2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 13.23
Output dim: 35, lower bound: -5.2118370, upper bound: 5.2118376

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -57.7383842, -32.6465416, -57.6057587, -32.6330605, -17.5741959, 17.4286957
1: -39.2681389, -20.2209091, -39.1603661, -20.2117367, -11.9336090, 11.8122902
2: -27.2873650, -11.1746340, -27.2004509, -11.1682844, -11.0512695, 10.9501915
3: -31.5990391, -14.0924969, -31.5504208, -14.0900068, -10.9619179, 10.9120560
4: -29.4576874, -8.6546593, -29.3457909, -8.6454525, -14.2170677, 14.1028824
5: -31.7686749, -13.5528374, -31.7204132, -13.5457649, -12.1679230, 12.1119881
6: -14.8665733, 2.9614453, -14.8814163, 2.8757820, -11.5189247, 11.6603432
7: -46.7097588, -25.5534134, -46.6064682, -25.5437469, -12.0587883, 11.9343643
8: -41.5373344, -19.8619919, -41.4199562, -19.8489132, -10.7627831, 10.6266613
9: -24.3131142, -5.1463799, -24.2324791, -5.1404390, -16.4891129, 16.3899384
10: -52.1483536, -29.6687412, -52.0211563, -29.6655960, -17.0870667, 16.9995804
11: -47.9118462, -27.0765305, -47.8873253, -27.0955009, -14.9795227, 15.0735168
12: -13.3374510, 5.9155293, -13.3300505, 5.8956714, -15.3690186, 15.3950348
13: -9.2932825, 9.7303257, -9.2106152, 9.7369795, -16.3027725, 16.2378120
14: -86.2496414, -59.5367584, -86.0681915, -59.5121841, -19.9991531, 19.7351112
15: -29.6292515, -11.9328537, -29.5475540, -11.9270725, -12.1907120, 12.0953178
16: -43.3893967, -22.5748100, -43.3090553, -22.5674267, -16.2162628, 16.1454544
17: -100.0344086, -70.0493240, -99.8823547, -70.0333786, -22.0450439, 22.0406952
18: -17.7528000, 3.4554203, -17.7441940, 3.4092367, -13.6306381, 13.6707191
19: -20.9996662, -6.4163527, -20.9710693, -6.4453545, -12.3657303, 12.4063187
20: -8.1811714, 5.6250582, -8.1765270, 5.5644693, -13.7456408, 13.8015852
21: -30.4730492, -12.1043873, -30.4319897, -12.1468277, -16.0307541, 16.1050034
22: -24.7909241, -8.3443050, -24.7690811, -8.3613911, -12.1070328, 12.1562843
23: -16.8807640, 0.1716345, -16.8661194, 0.1451559, -14.0864868, 14.1530876
24: -8.0026102, 6.9124637, -8.0026283, 6.8576117, -12.7151031, 12.7975807
25: -4.5790124, 11.7808504, -4.5746460, 11.7182198, -14.1294327, 14.2062492
26: -23.0428905, -1.5613728, -23.0357018, -1.6018660, -18.2152786, 18.3120956
27: -17.7902679, -3.7959993, -17.7906990, -3.8414397, -12.8182907, 12.8870392
28: -3.3147287, 16.2023544, -3.3138797, 16.1388550, -15.9320831, 15.9902191
29: -41.7153664, -23.3447647, -41.6933441, -23.3553238, -14.4879456, 14.5343628
30: -11.7779064, 7.2850733, -11.7753563, 7.1820059, -17.6453552, 17.7648010
31: -22.8766785, -4.3443851, -22.8581467, -4.3947320, -15.2188950, 15.2493286
32: -3.7782085, 10.6121845, -3.7761161, 10.5852604, -11.1788406, 11.2062912
33: 10.5329227, 30.9607830, 10.5293598, 30.8820419, -16.2191086, 16.3102417
34: 11.2926693, 29.0071945, 11.2845564, 28.8966827, -11.2817078, 11.4151115
35: 22.9411278, 40.5476913, 22.9354630, 40.4552307, -11.2528648, 11.3177071
36: 17.9712658, 34.5613861, 17.9600334, 34.4910851, -12.2813492, 12.3747330
37: 7.8852215, 28.1256981, 7.8853493, 28.1009045, -16.7118683, 16.7139473
38: 6.6123581, 26.5868702, 6.6000261, 26.5357819, -14.3170319, 14.3676910
39: 5.7299600, 25.9474869, 5.7429757, 25.9399529, -16.1840820, 16.1809387
40: 0.6056595, 19.8843994, 0.6013966, 19.8236580, -12.5026588, 12.5859375
41: -4.0684466, 9.1059313, -4.0712156, 9.0735855, -10.9217567, 10.9701691
42: -27.5605240, -10.8265190, -27.5612545, -10.8610334, -11.4764519, 11.5302963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=110, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 739

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1684369, upper bound: 5.2113080
time: 7.50 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1806779, upper bound: 5.2113553
time: 6.98 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -57.7426910, -32.6465836, -57.6169891, -32.6151428, -17.5873833, 17.4353752
1: -39.2722626, -20.2212486, -39.1696358, -20.1949844, -11.9501801, 11.8159981
2: -27.2885380, -11.1744556, -27.2039719, -11.1598740, -11.0610809, 10.9524002
3: -31.5995636, -14.0960007, -31.5517273, -14.0914621, -10.9655190, 10.9186401
4: -29.4617844, -8.6544342, -29.3552895, -8.6151295, -14.2486954, 14.1101189
5: -31.7688255, -13.5527449, -31.7220478, -13.5383282, -12.1686401, 12.1274796
6: -14.8664284, 2.9609241, -14.8811207, 2.8759623, -11.5203705, 11.6617851
7: -46.7123947, -25.5542545, -46.6127930, -25.5401802, -12.0632973, 11.9370880
8: -41.5373268, -19.8622093, -41.4202576, -19.8460045, -10.7748146, 10.6234303
9: -24.3205338, -5.1463184, -24.2511082, -5.0941582, -16.5374222, 16.4026642
10: -52.1581573, -29.6680660, -52.0424042, -29.6022568, -17.1550674, 17.0124512
11: -47.9130402, -27.0837898, -47.8849716, -27.1055984, -14.9769287, 15.1043816
12: -13.3397770, 5.9172239, -13.3418102, 5.9219389, -15.3976212, 15.4078331
13: -9.3021278, 9.7307196, -9.2342014, 9.7802515, -16.3519592, 16.2518845
14: -86.2513351, -59.5390091, -86.0765915, -59.5158577, -19.9928436, 19.7724228
15: -29.6301765, -11.9324570, -29.5505791, -11.9127626, -12.2009735, 12.0954018
16: -43.3996086, -22.5757484, -43.3346405, -22.5213032, -16.2542038, 16.1616287
17: -100.0456009, -70.0501022, -99.9087601, -70.0146332, -22.0448456, 22.0827637
18: -17.7540703, 3.4573350, -17.7634506, 3.4163837, -13.6366043, 13.7142334
19: -21.0026093, -6.4164343, -20.9783020, -6.4215307, -12.3901596, 12.4096222
20: -8.1815205, 5.6260414, -8.1883707, 5.5683932, -13.7499142, 13.8144121
21: -30.4788475, -12.1044655, -30.4450970, -12.1274805, -16.0370789, 16.1262779
22: -24.7920685, -8.3443222, -24.7733421, -8.3453093, -12.1220360, 12.1634369
23: -16.8813190, 0.1728346, -16.8824368, 0.1498399, -14.0892029, 14.1761818
24: -8.0029240, 6.9213161, -8.0413580, 6.8765769, -12.7277985, 12.8382263
25: -4.5795798, 11.7816448, -4.5840101, 11.7213717, -14.1269455, 14.2299652
26: -23.0427818, -1.5624676, -23.0483704, -1.6001506, -18.2219772, 18.3307037
27: -17.7909851, -3.7875655, -17.8321304, -3.8227916, -12.8312836, 12.9284744
28: -3.3152101, 16.2041912, -3.3402207, 16.1441307, -15.9374390, 16.0138245
29: -41.7181702, -23.3446484, -41.6998672, -23.3398476, -14.4803314, 14.5523224
30: -11.7792530, 7.2971497, -11.8362579, 7.2094212, -17.6641006, 17.8249512
31: -22.8790073, -4.3446026, -22.8643608, -4.3702545, -15.2486191, 15.2543755
32: -3.7827106, 10.6143208, -3.7893295, 10.6132221, -11.2224045, 11.2177124
33: 10.5328026, 30.9605713, 10.5188961, 30.8861542, -16.2425995, 16.3151169
34: 11.2918777, 29.0198269, 11.2217894, 28.9223442, -11.2985992, 11.5048561
35: 22.9404640, 40.5533752, 22.8959007, 40.4673080, -11.2588387, 11.3629799
36: 17.9722404, 34.5666504, 17.9307117, 34.5014610, -12.2848473, 12.4034233
37: 7.8853154, 28.1266575, 7.8694901, 28.1036148, -16.7192001, 16.7225914
38: 6.6127987, 26.5892715, 6.5676918, 26.5464668, -14.3261185, 14.4103165
39: 5.7289863, 25.9478607, 5.7283435, 25.9724579, -16.2214661, 16.1946411
40: 0.6053953, 19.8874016, 0.5783062, 19.8303146, -12.5322571, 12.5829735
41: -4.0703735, 9.1075611, -4.0772891, 9.0842361, -10.9452171, 10.9714012
42: -27.5631485, -10.8255434, -27.5665016, -10.8510704, -11.4967079, 11.5289688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=110, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1353
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1496
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 739

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1753989, upper bound: 5.2113080
time: 9.60 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1876320, upper bound: 5.2113553
time: 5.80 seconds

## BFS IS instance: IS_B2_A2_A2_A2_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -57.8136673, -32.6166306, -57.6165543, -32.6045532, -17.6696815, 17.4602127
1: -39.3218384, -20.1991806, -39.1697426, -20.1874084, -12.0070305, 11.8339233
2: -27.3307877, -11.1548243, -27.2036667, -11.1535854, -11.1095314, 10.9677696
3: -31.6118984, -14.0835657, -31.5518761, -14.0879450, -10.9800110, 10.9323196
4: -29.5227814, -8.6295443, -29.3553238, -8.6068487, -14.3180084, 14.1308250
5: -31.7890778, -13.5361938, -31.7221546, -13.5330086, -12.1950188, 12.1441650
6: -14.8905573, 3.0183153, -14.8890638, 2.8747501, -11.5327988, 11.7272835
7: -46.7489166, -25.5350533, -46.6128387, -25.5334625, -12.1072922, 11.9538956
8: -41.5926323, -19.8331890, -41.4203796, -19.8359776, -10.8401642, 10.6467056
9: -24.3655453, -5.1246929, -24.2509823, -5.0876102, -16.5888519, 16.4216919
10: -52.2130432, -29.6309624, -52.0423393, -29.5918598, -17.2259140, 17.0430756
11: -47.9255714, -27.0605278, -47.8867035, -27.1063881, -14.9812088, 15.1564331
12: -13.3489285, 5.9360785, -13.3430824, 5.9221783, -15.4155121, 15.4260635
13: -9.3552704, 9.7560596, -9.2367449, 9.7883186, -16.4131546, 16.2779846
14: -86.3761902, -59.4933929, -86.0758743, -59.4986992, -20.1344681, 19.8070450
15: -29.6764011, -11.9113922, -29.5509300, -11.9059296, -12.2540169, 12.1128616
16: -43.4357224, -22.5542240, -43.3351021, -22.5145836, -16.3019409, 16.1818924
17: -100.1067276, -70.0217743, -99.9091797, -70.0048065, -22.0711975, 22.1000214
18: -17.7650757, 3.4661531, -17.7644539, 3.4169395, -13.6421204, 13.7269325
19: -21.0171394, -6.4042096, -20.9817142, -6.4213457, -12.3992691, 12.4383965
20: -8.2011747, 5.6486444, -8.1931448, 5.5681362, -13.7693110, 13.8417892
21: -30.4966202, -12.0871353, -30.4476738, -12.1279612, -16.0460587, 16.1721077
22: -24.8084221, -8.3299065, -24.7765999, -8.3453064, -12.1332169, 12.1966019
23: -16.8950005, 0.1906457, -16.8854446, 0.1502484, -14.0939636, 14.2166595
24: -8.0190487, 6.9471202, -8.0459414, 6.8767700, -12.7385750, 12.8805656
25: -4.5991521, 11.8109531, -4.5896082, 11.7215033, -14.1453171, 14.2667847
26: -23.0636349, -1.5466583, -23.0532990, -1.5996146, -18.2304535, 18.3876190
27: -17.8093548, -3.7593255, -17.8372898, -3.8227196, -12.8458557, 12.9678307
28: -3.3389268, 16.2383270, -3.3466768, 16.1446075, -15.9612732, 16.0538559
29: -41.7324371, -23.3237991, -41.7023239, -23.3400803, -14.4932938, 14.5850296
30: -11.8039474, 7.3536386, -11.8434315, 7.2094541, -17.6862640, 17.8907547
31: -22.8982811, -4.3266926, -22.8695221, -4.3700919, -15.2675323, 15.2791328
32: -3.7979269, 10.6344433, -3.7933912, 10.6126995, -11.2322884, 11.2402840
33: 10.4926987, 31.0122738, 10.5068779, 30.8862457, -16.2785416, 16.3781967
34: 11.2678547, 29.0647945, 11.2144985, 28.9224243, -11.3188324, 11.5568771
35: 22.9090309, 40.5864067, 22.8865967, 40.4670868, -11.2840919, 11.4018364
36: 17.9421597, 34.5937271, 17.9211807, 34.5013428, -12.3100319, 12.4406548
37: 7.8619895, 28.1394310, 7.8636694, 28.1033669, -16.7441940, 16.7405624
38: 6.5880141, 26.5966682, 6.5606766, 26.5453682, -14.3539581, 14.4291458
39: 5.6979899, 25.9497719, 5.7200856, 25.9694309, -16.2538834, 16.2105484
40: 0.5846176, 19.9172821, 0.5726728, 19.8304291, -12.5517044, 12.6156082
41: -4.0857658, 9.1428442, -4.0818653, 9.0836182, -10.9581108, 11.0106888
42: -27.5807915, -10.7785778, -27.5720520, -10.8517389, -11.5094757, 11.5825157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=110, inp2_unstable=113, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=124, inp2_unstable=124, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1326
type: B, layer: 1, pos: 1326
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1342
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1342
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1353
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1327
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1327
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1358
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1358
type: A, layer: 1, pos: 1351
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1452
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1359

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 739

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1883187, upper bound: 5.2113142
time: 5.43 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2037561, upper bound: 5.2113646
time: 5.55 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 13.08 seconds
IS_B2_A2_A2_A2_A2_B1_B1_A1_A2_A1, status: Status.VERIFIED, split count: 10, time: 13.08
Output dim: 35, lower bound: -5.1684369, upper bound: 5.2113080
IS_B2_A2_A2_A2_A2_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 10, time: 13.08
Output dim: 35, lower bound: -5.1806779, upper bound: 5.2113553
IS_B2_A2_A2_A2_A2_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 10, time: 13.08
Output dim: 35, lower bound: -5.1753989, upper bound: 5.2113080
IS_B2_A2_A2_A2_A2_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 10, time: 13.08
Output dim: 35, lower bound: -5.1876320, upper bound: 5.2113553
IS_B2_A2_A2_A2_A2_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 10, time: 13.08
Output dim: 35, lower bound: -5.1883187, upper bound: 5.2113142
IS_B2_A2_A2_A2_A2_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 10, time: 13.08
Output dim: 35, lower bound: -5.2037561, upper bound: 5.2113646
IS_B2_A2_A2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 13.08
Output dim: 35, lower bound: -5.1869036, upper bound: 5.2118326
IS_B2_A2_A2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 13.08
Output dim: 35, lower bound: -5.2030181, upper bound: 5.2118376
IS_B2_A2_A2_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 13.08
Output dim: 35, lower bound: -5.1957330, upper bound: 5.2118325
IS_B2_A2_A2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 13.08
Output dim: 35, lower bound: -5.2118370, upper bound: 5.2118376

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 21.78 + 1787.65 = 1809.43 seconds
