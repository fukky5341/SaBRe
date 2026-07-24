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
execution time: IAR + RelationalAnalysis = 2.62 + 19.04 = 21.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 35, lower bound: -5.2169764, upper bound: 5.2169764

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1358

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 639

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2121837, upper bound: 5.2116575
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2116575, upper bound: 5.2121837
time: 12.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.80
Output dim: 35, lower bound: -5.2121837, upper bound: 5.2116575
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.80
Output dim: 35, lower bound: -5.2116575, upper bound: 5.2121837

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5708313, 17.5732689
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.9044838, 11.9067879
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0282173, 11.0286827
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9659386, 10.9683113
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2106400, 14.2115707
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1763916, 12.1788788
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6562042, 11.6580353
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0444832, 12.0470161
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7329025, 10.7362595
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4949265, 16.4962158
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1785202, 17.1793365
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0818787, 15.0820351
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4553452, 15.4549789
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3224487, 16.3228760
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9788818, 19.9801178
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1687202, 12.1694984
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2638016, 16.2650375
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1865158, 22.1866684
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7064972, 13.7046432
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4394150, 12.4381943
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1385193, 16.1395073
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1722946, 12.1720467
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1723557, 14.1713600
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7915955, 12.7899895
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1914978, 14.1914215
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3231049, 18.3222580
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9046936, 12.9048500
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9844513, 15.9829788
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5519104, 14.5519714
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7511139, 17.7515869
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2879639, 15.2864723
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2884369, 11.2883606
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3235855, 16.3228378
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4503670, 11.4496689
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3720932, 11.3707962
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4087791, 12.4080696
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7837219, 16.7833099
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4383316, 14.4380913
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2399902, 16.2389374
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6422653, 12.6421242
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9801712, 10.9802513
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5337486, 11.5361404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1696

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1322

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2119675, upper bound: 5.2111533
time: 19.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2116795, upper bound: 5.2114413
time: 6.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5712433, 17.5708351
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.9048615, 11.9044838
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0282974, 11.0282173
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9663391, 10.9659386
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2107773, 14.2106361
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1767807, 12.1763878
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6564217, 11.6562042
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0449066, 12.0444832
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7334137, 10.7329025
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4951477, 16.4949265
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1786499, 17.1785202
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0818787, 15.0818787
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4549789, 15.4550285
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3225098, 16.3224487
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9789429, 19.9788818
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1688499, 12.1687164
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2640076, 16.2637978
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1865158, 22.1865158
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7046509, 13.7049484
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4381943, 12.4383965
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1385345, 16.1385307
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1720428, 12.1721611
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1713562, 14.1715317
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7899933, 12.7902565
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1914215, 14.1914444
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3222580, 18.3224030
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9048538, 12.9048805
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9829788, 15.9832306
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5519714, 14.5519714
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7511139, 17.7511139
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2864685, 15.2867241
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2883606, 11.2883644
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3228378, 16.3229675
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4496689, 11.4497757
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3707962, 11.3710098
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4080696, 12.4081612
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7833099, 16.7833786
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4383469, 14.4383278
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2389374, 16.2389450
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6421280, 12.6421318
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9802513, 10.9802513
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5340080, 11.5337486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1349

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1433

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2071981, upper bound: 5.2119135
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2113873, upper bound: 5.2077244
time: 29.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 37.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 37.91
Output dim: 35, lower bound: -5.2119675, upper bound: 5.2111533
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 37.91
Output dim: 35, lower bound: -5.2116795, upper bound: 5.2114413
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 37.91
Output dim: 35, lower bound: -5.2071981, upper bound: 5.2119135
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 37.91
Output dim: 35, lower bound: -5.2113873, upper bound: 5.2077244

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5703812, 17.5727310
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.9043655, 11.9066658
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0280151, 11.0284920
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9658318, 10.9682274
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2102966, 14.2112503
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1761780, 12.1786957
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6560745, 11.6579361
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0443344, 12.0469208
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7326851, 10.7360516
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4948959, 16.4961700
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1784859, 17.1792755
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0817261, 15.0818977
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4552193, 15.4548607
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3216248, 16.3221207
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9785156, 19.9796524
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1683311, 12.1690292
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2635880, 16.2648773
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1863480, 22.1864929
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7060394, 13.7041054
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4391937, 12.4379311
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1382446, 16.1392899
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1721954, 12.1719360
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1719971, 14.1709290
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7913513, 12.7897034
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1914444, 14.1913567
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3227005, 18.3218307
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9039383, 12.9041138
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9844437, 15.9829712
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5517578, 14.5518112
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7510605, 17.7515564
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2877808, 15.2863617
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2880325, 11.2880096
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3230896, 16.3222504
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4492645, 11.4484367
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3719406, 11.3706474
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4087563, 12.4080505
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7831345, 16.7825546
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4381104, 14.4377861
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2399521, 16.2389069
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6410751, 12.6408882
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9798889, 10.9799194
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5331955, 11.5354881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1351

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1743

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2112391, upper bound: 5.2072662
time: 13.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2080806, upper bound: 5.2104252
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5591888, 17.5580559
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8986778, 11.8975220
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0258713, 11.0258980
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9626389, 10.9626999
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2001381, 14.2009010
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1689453, 12.1690102
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6439705, 11.6430435
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0457306, 12.0450211
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7333260, 10.7328243
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4934006, 16.4927063
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1806908, 17.1801910
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0577850, 15.0561028
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4553871, 15.4551697
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3149643, 16.3153648
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9591522, 19.9573975
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1496010, 12.1518555
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2639389, 16.2630157
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1918030, 22.1897125
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6946526, 13.6942215
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4357758, 12.4356575
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1387253, 16.1386490
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1772461, 12.1774979
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1690521, 14.1688309
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7899017, 12.7901649
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1919937, 14.1928253
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3268204, 18.3264236
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9014053, 12.9012527
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9825134, 15.9826660
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5555496, 14.5545349
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7441483, 17.7436676
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2856598, 15.2857246
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2786980, 11.2782402
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3215103, 16.3217621
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4467087, 11.4472046
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3712807, 11.3715591
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4057388, 12.4061050
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7674789, 16.7686691
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4385414, 14.4385719
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2257156, 16.2267456
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6037064, 12.6037102
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9664764, 10.9646149
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5212898, 11.5203018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1342

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1479

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2067941, upper bound: 5.2116029
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2068876, upper bound: 5.2115094
time: 5.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.70
Output dim: 35, lower bound: -5.2112391, upper bound: 5.2072662
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.70
Output dim: 35, lower bound: -5.2080806, upper bound: 5.2104252
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.70
Output dim: 35, lower bound: -5.2067941, upper bound: 5.2116029
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.70
Output dim: 35, lower bound: -5.2068876, upper bound: 5.2115094

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 21.65 + 116.95 = 138.60 seconds
