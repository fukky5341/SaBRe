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
execution time: IAR + RelationalAnalysis = 2.46 + 19.10 = 21.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 35, lower bound: -5.2169764, upper bound: 5.2169764

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 721

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1955169, upper bound: 5.2160960
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2160960, upper bound: 5.1955169
time: 5.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.14
Output dim: 35, lower bound: -5.1955169, upper bound: 5.2160960
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.14
Output dim: 35, lower bound: -5.2160960, upper bound: 5.1955169

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5651245, 17.5597763
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8999748, 11.8952904
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0247726, 11.0216179
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9639931, 10.9619255
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2058716, 14.2015915
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1748962, 12.1732178
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6463242, 11.6510277
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0404015, 12.0349159
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7271805, 10.7203236
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4929352, 16.4910202
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1731758, 17.1684799
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0836639, 15.0809174
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4532852, 15.4542236
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3217087, 16.3208694
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9692688, 19.9589424
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1652222, 12.1620598
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2631149, 16.2617683
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1825409, 22.1734619
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7045860, 13.7049255
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4387283, 12.4380417
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1393738, 16.1379433
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1720581, 12.1708260
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1713562, 14.1702881
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7911873, 12.7899055
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1914368, 14.1914368
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3235016, 18.3216400
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9055862, 12.9044991
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9815369, 15.9824219
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5527802, 14.5509529
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7511139, 17.7510300
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2860031, 15.2862053
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2839546, 11.2863312
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3153992, 16.3190155
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4447327, 11.4475899
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3623695, 11.3671455
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4009476, 12.4048233
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7803726, 16.7812958
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4349899, 14.4372292
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2369995, 16.2375107
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6338882, 12.6375122
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9768562, 10.9785538
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5327415, 11.5333481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1767130, upper bound: 5.2156892
time: 16.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1951196, upper bound: 5.1972953
time: 10.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5597687, 17.5651169
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8952904, 11.8999748
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0216179, 11.0247726
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9619255, 10.9639931
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.2015839, 14.2058754
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1732178, 12.1748962
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6510277, 11.6463242
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0349159, 12.0404015
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.7203217, 10.7271843
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4910202, 16.4929428
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1684837, 17.1731796
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0809174, 15.0836639
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4542160, 15.4532890
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3208694, 16.3217087
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.9589386, 19.9692726
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1620598, 12.1652222
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2617722, 16.2631149
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1734467, 22.1825409
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7049217, 13.7045860
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4380417, 12.4387245
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1379395, 16.1393700
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1708221, 12.1720581
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1702881, 14.1713524
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7899055, 12.7911911
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1914368, 14.1914368
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3216400, 18.3235016
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9045029, 12.9055824
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9824142, 15.9815369
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5509491, 14.5527840
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7510300, 17.7511139
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2862015, 15.2860107
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2863312, 11.2839546
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.3190155, 16.3153992
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4475899, 11.4447327
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3671455, 11.3623657
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.4048233, 12.4009438
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7812881, 16.7803802
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4372330, 14.4349861
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2375107, 16.2369995
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6375122, 12.6338844
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9785538, 10.9768562
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5333481, 11.5327415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1972953, upper bound: 5.1951196
time: 24.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2156892, upper bound: 5.1767130
time: 26.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 53.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 53.83
Output dim: 35, lower bound: -5.1767130, upper bound: 5.2156892
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 53.83
Output dim: 35, lower bound: -5.1951196, upper bound: 5.1972953
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 53.83
Output dim: 35, lower bound: -5.1972953, upper bound: 5.1951196
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 53.83
Output dim: 35, lower bound: -5.2156892, upper bound: 5.1767130

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5283585, 17.5189476
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8755302, 11.8676758
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0130424, 11.0084534
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9641113, 10.9619255
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1858902, 14.1789627
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1739616, 12.1721992
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6052704, 11.6148109
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0113907, 12.0021400
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6901245, 10.6783237
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4784927, 16.4734421
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1103630, 17.0984421
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0801773, 15.0729866
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4536209, 15.4547081
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3198547, 16.3188705
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8866730, 19.8653717
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1589584, 12.1547203
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2470627, 16.2420425
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1377182, 22.1212769
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7082138, 13.7092171
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4319000, 12.4285126
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1324615, 16.1272087
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1731682, 12.1707420
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1529465, 14.1486397
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7941780, 12.7912560
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1872635, 14.1861267
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3386765, 18.3340073
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9111099, 12.9086647
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9810791, 15.9820480
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5532761, 14.5500832
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7532578, 17.7520294
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2821198, 15.2808762
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2688828, 11.2730484
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2792587, 16.2869110
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4166603, 11.4227524
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3199577, 11.3298759
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3682556, 12.3759918
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7811966, 16.7823334
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4188499, 14.4230804
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2401123, 16.2411804
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6092415, 12.6157875
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9665031, 10.9694710
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5320435, 11.5326691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1604974, upper bound: 5.2152052
time: 11.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1762180, upper bound: 5.1994877
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5189438, 17.5283546
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8676758, 11.8755302
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0084534, 11.0130424
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9619255, 10.9641113
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1789627, 14.1858864
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1721992, 12.1739655
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6148109, 11.6052704
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0021400, 12.0113907
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6783218, 10.6901264
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4734421, 16.4784927
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0984383, 17.1103668
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0729904, 15.0801773
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4547043, 15.4536209
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3188705, 16.3198586
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8653717, 19.8866730
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1547203, 12.1589584
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2420425, 16.2470665
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1212692, 22.1377258
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7092209, 13.7082176
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4285126, 12.4319038
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1272125, 16.1324654
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1707420, 12.1731682
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1486359, 14.1529465
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7912483, 12.7941742
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1861343, 14.1872597
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3340073, 18.3386765
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9086685, 12.9111099
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9820480, 15.9810791
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5500870, 14.5532761
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7520294, 17.7532578
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2808838, 15.2821236
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2730484, 11.2688866
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2869110, 16.2792587
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4227524, 11.4166603
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3298759, 11.3199577
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3759918, 12.3682518
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7823334, 16.7811966
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4230766, 14.4188576
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2411804, 16.2401123
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6157875, 12.6092415
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9694710, 10.9665031
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5326691, 11.5320435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1994877, upper bound: 5.1762180
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2152052, upper bound: 5.1604974
time: 5.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 35, lower bound: -5.1604974, upper bound: 5.2152052
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.26
Output dim: 35, lower bound: -5.1762180, upper bound: 5.1994877
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.26
Output dim: 35, lower bound: -5.1994877, upper bound: 5.1762180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.26
Output dim: 35, lower bound: -5.2152052, upper bound: 5.1604974

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5219269, 17.5066910
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8709755, 11.8589897
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0097656, 11.0022087
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9626961, 10.9592323
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1808968, 14.1694565
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1731873, 12.1707115
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5898552, 11.6044350
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0071068, 11.9939728
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6830788, 10.6648750
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4749222, 16.4666367
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.1030121, 17.0844193
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0822601, 15.0721817
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4531784, 15.4547081
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3184128, 16.3170700
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8730927, 19.8394699
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1558151, 12.1487198
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2449341, 16.2379913
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1387024, 22.1144791
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7079620, 13.7080040
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4329071, 12.4282036
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1343384, 16.1266060
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1743546, 12.1706238
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1507034, 14.1479340
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7940216, 12.7907944
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1871948, 14.1861191
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3420029, 18.3336411
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9116592, 12.9086151
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9801025, 15.9815369
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5536041, 14.5499268
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7524185, 17.7512283
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2821426, 15.2808914
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2626495, 11.2697563
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2689667, 16.2815933
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4115372, 11.4200668
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3124886, 11.3262825
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3611298, 12.3722534
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7786865, 16.7812920
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4182396, 14.4226494
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2396469, 16.2409439
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5990028, 12.6104126
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9607811, 10.9664726
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5272293, 11.5290298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 737

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1465442, upper bound: 5.2149004
time: 15.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1594442, upper bound: 5.1916657
time: 10.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5066833, 17.5219269
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8589897, 11.8709755
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0022087, 11.0097656
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9592323, 10.9626961
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1694527, 14.1809044
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1707153, 12.1731873
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.6044350, 11.5898552
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9939728, 12.0071068
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6648750, 10.6830788
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4666367, 16.4749222
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0844193, 17.1030121
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0721817, 15.0822601
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4547043, 15.4531822
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3170700, 16.3184128
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8394699, 19.8730927
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1487198, 12.1558151
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2379913, 16.2449341
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1144714, 22.1387100
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7080078, 13.7079659
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4282074, 12.4328995
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1266022, 16.1343422
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1706238, 12.1743584
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1479340, 14.1506996
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7907944, 12.7940216
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1861115, 14.1871948
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3336411, 18.3420029
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9086151, 12.9116554
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9815369, 15.9801025
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5499268, 14.5536079
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7512283, 17.7524185
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2808914, 15.2821426
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2697563, 11.2626495
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2815933, 16.2689667
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4200668, 11.4115334
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3262825, 11.3124886
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3722534, 12.3611259
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7812958, 16.7786903
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4226494, 14.4182358
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2409439, 16.2396469
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6104164, 12.5990028
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9664726, 10.9607811
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5290298, 11.5272293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 737

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1916657, upper bound: 5.1594442
time: 24.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2149004, upper bound: 5.1465442
time: 5.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 32.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 32.16
Output dim: 35, lower bound: -5.1465442, upper bound: 5.2149004
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 32.16
Output dim: 35, lower bound: -5.1594442, upper bound: 5.1916657
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 32.16
Output dim: 35, lower bound: -5.1916657, upper bound: 5.1594442
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 32.16
Output dim: 35, lower bound: -5.2149004, upper bound: 5.1465442

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5130692, 17.4950409
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8640900, 11.8495712
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0057220, 10.9968109
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9622917, 10.9587326
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1750870, 14.1615028
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1731033, 12.1706047
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5752182, 11.5935135
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9990425, 11.9829369
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6746902, 10.6534004
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4685059, 16.4578552
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0848045, 17.0595093
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0822601, 15.0680199
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4487724, 15.4506302
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3167191, 16.3148994
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8478546, 19.8049469
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1535187, 12.1457672
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2362289, 16.2254639
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1196899, 22.0875626
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7074127, 13.7077751
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4305878, 12.4231644
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1311111, 16.1197319
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1730690, 12.1674271
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1458969, 14.1397972
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7938652, 12.7892990
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1861420, 14.1839333
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3428955, 18.3317490
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9118652, 12.9076691
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9799500, 15.9814224
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5526123, 14.5471725
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7522202, 17.7502518
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2798080, 15.2768822
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2582130, 11.2664757
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2552719, 16.2715759
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3999786, 11.4116211
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2970276, 11.3149796
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3495941, 12.3638229
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7763214, 16.7794456
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4103203, 14.4168587
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2388306, 16.2403641
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5856819, 12.6005440
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9566383, 10.9634094
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5259171, 11.5279808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1441335, upper bound: 5.2136398
time: 17.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1452015, upper bound: 5.2125453
time: 9.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4950485, 17.5130653
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8495712, 11.8640900
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9968109, 11.0057220
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9587326, 10.9622917
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1614990, 14.1750870
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1706085, 12.1731033
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5935135, 11.5752182
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9829407, 11.9990425
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6534004, 10.6746902
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4578552, 16.4685059
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0595131, 17.0848083
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0680237, 15.0822601
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4506340, 15.4487762
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3149033, 16.3167191
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8049469, 19.8478546
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1457634, 12.1535187
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2254639, 16.2362289
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0875549, 22.1196823
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7077789, 13.7074089
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4231644, 12.4305840
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1197281, 16.1311150
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1674309, 12.1730690
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1397934, 14.1458969
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7893028, 12.7938614
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1839294, 14.1861382
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3317490, 18.3428879
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9076691, 12.9118652
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9814224, 15.9799500
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5471725, 14.5526123
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7502518, 17.7522202
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2768784, 15.2798004
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2664795, 11.2582130
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2715759, 16.2552719
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4116211, 11.3999825
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3149796, 11.2970276
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3638229, 12.3495941
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7794495, 16.7763100
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4168587, 14.4103203
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2403641, 16.2388306
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6005440, 12.5856895
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9634094, 10.9566383
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5279770, 11.5259171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2125453, upper bound: 5.1452015
time: 17.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2136398, upper bound: 5.1441335
time: 10.04 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 29.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.48
Output dim: 35, lower bound: -5.1441335, upper bound: 5.2136398
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.48
Output dim: 35, lower bound: -5.1452015, upper bound: 5.2125453
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.48
Output dim: 35, lower bound: -5.2125453, upper bound: 5.1452015
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.48
Output dim: 35, lower bound: -5.2136398, upper bound: 5.1441335

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5143013, 17.4928894
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8655281, 11.8466949
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0080757, 10.9927139
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9654274, 10.9537239
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1768417, 14.1580162
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1752434, 12.1681099
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5731049, 11.5940018
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -12.0009384, 11.9796219
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6789169, 10.6460381
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4684677, 16.4571991
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0846291, 17.0594254
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0813904, 15.0682068
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4486504, 15.4545479
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3176041, 16.3133698
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8493118, 19.8005142
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1543617, 12.1441040
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2358551, 16.2274933
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1193237, 22.0861664
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7048111, 13.7092552
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4276581, 12.4248352
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1289062, 16.1195450
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1706886, 12.1686325
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1455536, 14.1399918
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7930565, 12.7897072
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1833496, 14.1855278
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3426743, 18.3316116
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9114838, 12.9078865
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9781418, 15.9824600
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5503769, 14.5483093
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7505722, 17.7510681
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2756042, 15.2792931
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2577705, 11.2672005
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2535477, 16.2722702
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3992767, 11.4113426
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2969551, 11.3149490
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3494644, 12.3637581
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7741089, 16.7821121
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4113235, 14.4147682
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2387619, 16.2400055
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5844231, 12.6009903
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9562073, 10.9639282
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5258217, 11.5279694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 754

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1338037, upper bound: 5.2132923
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1429051, upper bound: 5.1941936
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5109138, 17.4950409
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8612137, 11.8495712
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0016251, 10.9968109
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9572830, 10.9587326
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1716003, 14.1615028
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1706047, 12.1706047
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5752182, 11.5914001
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9957275, 11.9829369
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6673279, 10.6534004
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4685059, 16.4578171
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0848045, 17.0593414
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0822601, 15.0671501
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4487724, 15.4505043
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3151855, 16.3148994
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8434296, 19.8049469
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1518593, 12.1457672
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2362289, 16.2250900
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1182709, 22.0875626
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7074127, 13.7051811
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4305878, 12.4202423
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1311111, 16.1175232
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1730690, 12.1650429
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1458969, 14.1394577
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7938652, 12.7884941
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1861420, 14.1811485
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3427582, 18.3317490
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9118652, 12.9072876
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9799500, 15.9796143
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5526123, 14.5449371
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7522202, 17.7486038
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2798080, 15.2726822
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2582130, 11.2660370
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2552719, 16.2698593
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3997040, 11.4116211
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2969971, 11.3149796
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3495255, 12.3638229
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7763214, 16.7772446
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4082260, 14.4168587
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2388306, 16.2403030
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5856819, 12.5992775
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9566383, 10.9629822
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5259056, 11.5279808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 754

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1348683, upper bound: 5.2122040
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1439731, upper bound: 5.1931026
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4962654, 17.5109138
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8510094, 11.8612137
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9991646, 11.0016251
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9618683, 10.9572868
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1632538, 14.1716042
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1727409, 12.1706047
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5914001, 11.5757065
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9848366, 11.9957275
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6576271, 10.6673279
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4578171, 16.4678497
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0593376, 17.0847168
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0671539, 15.0824471
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4505119, 15.4526939
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3157806, 16.3151855
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8064041, 19.8434219
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1466103, 12.1518555
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2250900, 16.2382584
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0872040, 22.1182785
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7051773, 13.7088890
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4202423, 12.4322548
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1175232, 16.1309280
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1650429, 12.1742744
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1394577, 14.1460915
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7884941, 12.7942657
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1811523, 14.1877327
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3315353, 18.3427582
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9072876, 12.9120789
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9796143, 15.9809875
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5449371, 14.5537491
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7486038, 17.7530365
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2726822, 15.2822113
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2660370, 11.2589378
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2698593, 16.2559662
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4109192, 11.3997002
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3149071, 11.2969971
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3636932, 12.3495255
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7772446, 16.7789764
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4178619, 14.4082260
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2403030, 16.2384720
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5992775, 12.5861359
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9629822, 10.9571571
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5278816, 11.5259056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 754

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1931026, upper bound: 5.1439731
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2122040, upper bound: 5.1348683
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4928932, 17.5130653
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8466949, 11.8640900
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9927139, 11.0057220
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9537277, 10.9622917
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1580200, 14.1750870
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1681099, 12.1731033
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5935135, 11.5731049
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9796219, 11.9990425
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6460381, 10.6746902
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4578552, 16.4684677
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0595131, 17.0846329
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0680237, 15.0813904
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4506340, 15.4486504
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3133698, 16.3167191
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8005219, 19.8478546
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1441040, 12.1535187
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2254639, 16.2358551
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0861664, 22.1196823
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7077789, 13.7048149
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4231644, 12.4276619
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1197281, 16.1289062
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1674309, 12.1706810
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1397934, 14.1455536
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7893028, 12.7930527
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1839294, 14.1833534
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3316116, 18.3428879
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9076691, 12.9114838
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9814224, 15.9781418
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5471725, 14.5503769
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7502518, 17.7505722
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2768784, 15.2756042
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2664795, 11.2577705
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2715759, 16.2535477
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4113464, 11.3999825
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3149490, 11.2970276
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3637543, 12.3495941
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7794495, 16.7741089
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4147720, 14.4103203
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2403641, 16.2387619
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.6005440, 12.5844231
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9634094, 10.9562073
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5279694, 11.5259171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 754

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1941936, upper bound: 5.1429051
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2132923, upper bound: 5.1338037
time: 18.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 27.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.1338037, upper bound: 5.2132923
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.1429051, upper bound: 5.1941936
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.1348683, upper bound: 5.2122040
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.1439731, upper bound: 5.1931026
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.1931026, upper bound: 5.1439731
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.2122040, upper bound: 5.1348683
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.1941936, upper bound: 5.1429051
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 27.69
Output dim: 35, lower bound: -5.2132923, upper bound: 5.1338037

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5064926, 17.4816589
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8599014, 11.8386002
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0046806, 10.9878311
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9651642, 10.9533463
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1706390, 14.1490974
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1749496, 12.1677017
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5579758, 11.5834808
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9950676, 11.9711761
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6697044, 10.6327934
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4635010, 16.4500275
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0735092, 17.0434341
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0827026, 15.0673141
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4485741, 15.4544983
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3169098, 16.3123970
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8303070, 19.7731705
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1504936, 12.1385460
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2320251, 16.2219391
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1166382, 22.0779266
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7048683, 13.7086029
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4286308, 12.4243050
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1304092, 16.1188812
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1716461, 12.1682167
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1462402, 14.1386032
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7940140, 12.7892990
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1838913, 14.1852608
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3448639, 18.3312454
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9124069, 12.9077301
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9776001, 15.9820786
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5507736, 14.5479927
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7509384, 17.7510071
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2758026, 15.2787933
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2520218, 11.2631989
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2439423, 16.2655945
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3930054, 11.4069824
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2863312, 11.3075638
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3409882, 12.3578644
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7736130, 16.7817841
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4078522, 14.4123535
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2375793, 16.2391586
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5754280, 12.5947342
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9509125, 10.9602432
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5220985, 11.5252571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1744

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1336311, upper bound: 5.2032038
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1259884, upper bound: 5.2132924
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.5031052, 17.4838066
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8555870, 11.8414726
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9982300, 10.9919319
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9570198, 10.9583588
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1654053, 14.1525764
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1703262, 12.1702003
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5600891, 11.5808830
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9898529, 11.9744949
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6581192, 10.6401558
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4635391, 16.4506531
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0736847, 17.0433426
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0835724, 15.0662575
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4486809, 15.4504547
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3144989, 16.3139267
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.8244171, 19.7776108
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1479912, 12.1402092
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2324066, 16.2195358
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1156006, 22.0793381
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7074547, 13.7045288
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4315529, 12.4197121
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1326218, 16.1168594
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1740341, 12.1646233
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1465836, 14.1380653
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7948227, 12.7880859
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1866684, 14.1808815
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3449478, 18.3313751
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9127884, 12.9071312
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9794235, 15.9792404
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5530090, 14.5446205
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7525864, 17.7485352
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2800140, 15.2721825
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2524757, 11.2620354
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2456665, 16.2631836
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3934326, 11.4072609
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2863731, 11.3075943
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3410492, 12.3579330
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7758179, 16.7769165
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4047546, 14.4144478
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2376480, 16.2394485
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5766945, 12.5930252
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9513397, 10.9592972
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5221825, 11.5252647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1744

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1346957, upper bound: 5.2021139
time: 19.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1270530, upper bound: 5.2122042
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4850388, 17.5031052
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8429146, 11.8555870
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9942818, 10.9982300
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9614868, 10.9570198
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1543350, 14.1654015
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1723328, 12.1703224
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5808830, 11.5605736
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9763870, 11.9898529
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6443825, 10.6581173
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4506531, 16.4628830
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0433426, 17.0736008
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0662613, 15.0837593
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4504509, 15.4526176
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3148117, 16.3144951
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7790604, 19.8244171
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1410484, 12.1479912
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2195358, 16.2344322
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0789795, 22.1156006
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7045326, 13.7089386
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4197121, 12.4332237
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1168594, 16.1324310
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1646271, 12.1752357
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1380615, 14.1467819
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7880859, 12.7952271
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1808853, 14.1882668
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3311615, 18.3449478
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9071350, 12.9130020
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9792404, 15.9804459
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5446167, 14.5541458
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7485352, 17.7534027
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2721863, 15.2824173
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2620354, 11.2531853
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2631836, 16.2463608
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4065552, 11.3934326
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3075218, 11.2863731
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3578033, 12.3410530
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7769165, 16.7784729
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4154510, 14.4047546
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2394562, 16.2372894
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5930214, 12.5771370
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9592972, 10.9518585
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5251732, 11.5221825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1744

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2122042, upper bound: 5.1270530
time: 30.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2021139, upper bound: 5.1346957
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4816513, 17.5052567
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8386002, 11.8584595
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9878311, 11.0023308
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9533463, 10.9620323
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1491013, 14.1688805
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1677017, 12.1728210
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5829964, 11.5579758
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9711761, 11.9931717
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6327934, 10.6654816
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4506912, 16.4635010
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0435181, 17.0735092
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0671310, 15.0827026
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4505577, 15.4485741
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3123932, 16.3160248
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7731781, 19.8288498
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1385460, 12.1496544
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2199097, 16.2320290
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0779266, 22.1170120
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7071266, 13.7048645
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4226341, 12.4286308
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1190720, 16.1304092
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1670074, 12.1716461
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1384048, 14.1462402
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7888870, 12.7940140
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1836624, 14.1838875
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3312454, 18.3450851
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9075165, 12.9124069
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9810638, 15.9776001
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5468597, 14.5507736
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7501831, 17.7509384
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2763824, 15.2758064
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2624893, 11.2520218
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2649002, 16.2439423
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.4069824, 11.3937073
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.3075638, 11.2864037
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3578644, 12.3411217
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7791290, 16.7736130
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4123535, 14.4068451
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2395172, 16.2375793
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5942879, 12.5754280
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9597244, 10.9509087
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5252571, 11.5221939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1744

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2132924, upper bound: 5.1259884
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2032038, upper bound: 5.1336311
time: 9.13 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 17.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.1336311, upper bound: 5.2032038
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.1259884, upper bound: 5.2132924
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.1346957, upper bound: 5.2021139
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.1270530, upper bound: 5.2122042
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.2122042, upper bound: 5.1270530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.2021139, upper bound: 5.1346957
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.2132924, upper bound: 5.1259884
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 17.85
Output dim: 35, lower bound: -5.2032038, upper bound: 5.1336311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4884644, 17.4612236
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8429947, 11.8194618
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -11.0005150, 10.9830856
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9653587, 10.9535141
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1624565, 14.1400070
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1733017, 12.1659470
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5463333, 11.5732079
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9739227, 11.9472313
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6479759, 10.6079960
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4510651, 16.4354324
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0559883, 17.0211487
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0760498, 15.0571404
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4501152, 15.4559631
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3160324, 16.3112259
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7929611, 19.7306519
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1512222, 12.1392593
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2236099, 16.2100830
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0890503, 22.0468063
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6994247, 13.7041054
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4225044, 12.4165192
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1212158, 16.1071167
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1693497, 12.1656380
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1362076, 14.1264687
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7939224, 12.7886467
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1800537, 14.1805840
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3451462, 18.3310089
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9092636, 12.9037628
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9773102, 15.9817886
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5457458, 14.5420303
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7464600, 17.7447891
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2757874, 15.2783279
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2459984, 11.2579269
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2243652, 16.2484283
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3748055, 11.3909073
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2589149, 11.2833900
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3222008, 12.3412857
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7730179, 16.7810783
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4026451, 14.4086151
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2387848, 16.2403107
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5619659, 12.5826454
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9502792, 10.9596443
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5263939, 11.5288086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 720

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1209760, upper bound: 5.2124618
time: 24.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1241644, upper bound: 5.1996550
time: 7.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4850845, 17.4633636
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8386841, 11.8223343
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9940605, 10.9871826
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9572182, 10.9585228
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1572075, 14.1434898
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1686630, 12.1684532
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5484467, 11.5706062
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9687080, 11.9505424
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6363869, 10.6153622
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4511032, 16.4360504
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0561485, 17.0210571
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0769272, 15.0560875
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4502220, 15.4519196
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3136215, 16.3127632
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7870712, 19.7350998
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1487160, 12.1409302
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2239761, 16.2076797
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0880127, 22.0482330
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7020187, 13.7000313
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4254341, 12.4119263
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1234283, 16.1050949
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1717377, 12.1620483
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1365356, 14.1259308
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7947235, 12.7874336
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1828461, 14.1761971
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3452301, 18.3311310
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9096451, 12.9031677
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9791336, 15.9789505
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5480042, 14.5386581
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7480927, 17.7423172
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2799988, 15.2717171
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2464485, 11.2567596
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2260818, 16.2460175
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3752327, 11.3911858
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2589569, 11.2834167
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3222618, 12.3413506
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7752228, 16.7762184
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.3995476, 14.4107094
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2388535, 16.2406006
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5632324, 12.5809326
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9507103, 10.9586945
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5264816, 11.5288200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 720

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1220354, upper bound: 5.2113750
time: 11.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1252257, upper bound: 5.1985781
time: 6.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4646072, 17.4850807
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8237724, 11.8386841
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9895401, 10.9940605
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9616585, 10.9572182
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1452446, 14.1572113
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1705856, 12.1686668
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5706062, 11.5489311
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9524422, 11.9687080
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6195831, 10.6363869
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4360504, 16.4504395
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0210609, 17.0560760
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0560913, 15.0771065
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4519234, 15.4541588
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3136368, 16.3136215
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7365417, 19.7870712
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1417656, 12.1487198
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2076797, 16.2260170
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0478516, 22.0880127
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7000275, 13.7035027
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4119225, 12.4270973
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1050949, 16.1232376
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1620483, 12.1729393
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1259308, 14.1367416
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7874374, 12.7951279
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1762009, 14.1844368
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3309250, 18.3452301
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9031677, 12.9098587
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9789505, 15.9801559
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5386581, 14.5491180
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7423172, 17.7489243
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2717209, 15.2824020
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2567596, 11.2471657
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2460175, 16.2267838
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3904800, 11.3752327
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2833481, 11.2589569
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3412285, 12.3222656
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7762222, 16.7778740
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4117088, 14.3995476
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2406006, 16.2384949
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5809326, 12.5636749
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9586945, 10.9512253
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5287247, 11.5264816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 720

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1985781, upper bound: 5.1252257
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2113750, upper bound: 5.1220354
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4612198, 17.4872246
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8194580, 11.8415565
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9830856, 10.9981575
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9535141, 10.9622269
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1400108, 14.1606941
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1659470, 12.1711693
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5727196, 11.5463333
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9472313, 11.9720230
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6079941, 10.6437531
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4360886, 16.4510651
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0212288, 17.0559845
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0569611, 15.0760498
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4520302, 15.4501152
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3112259, 16.3151627
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7306519, 19.7915192
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1392593, 12.1503868
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2080383, 16.2236137
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0468140, 22.0894394
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7026215, 13.6994247
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4148521, 12.4225044
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1073074, 16.1212158
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1644363, 12.1693459
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1262665, 14.1362038
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7882462, 12.7939148
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1789856, 14.1800575
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3310089, 18.3453522
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9035492, 12.9092636
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9807663, 15.9773178
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5409088, 14.5457458
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7439575, 17.7464600
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2759247, 15.2757874
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2572098, 11.2460022
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2477264, 16.2243652
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3909073, 11.3755074
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2833900, 11.2589874
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3412895, 12.3223305
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7784271, 16.7730141
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4086189, 14.4016380
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2406693, 16.2387848
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5821991, 12.5619621
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9591293, 10.9502792
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5288086, 11.5264893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 720

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1996550, upper bound: 5.1241644
time: 17.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2124618, upper bound: 5.1209760
time: 5.75 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 25.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.1209760, upper bound: 5.2124618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.1241644, upper bound: 5.1996550
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.1220354, upper bound: 5.2113750
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.1252257, upper bound: 5.1985781
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.1985781, upper bound: 5.1252257
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.2113750, upper bound: 5.1220354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.1996550, upper bound: 5.1241644
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 25.42
Output dim: 35, lower bound: -5.2124618, upper bound: 5.1209760

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4868240, 17.4569626
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8415947, 11.8156967
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9998360, 10.9813690
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9650192, 10.9529572
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1614075, 14.1373215
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1726761, 12.1647263
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5441170, 11.5723534
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9721794, 11.9425659
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6465492, 10.6041679
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4501572, 16.4329376
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0534248, 17.0142899
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0761948, 15.0538864
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4499168, 15.4556427
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3158035, 16.3106918
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7897415, 19.7220230
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1503487, 12.1379318
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2227249, 16.2061462
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0856171, 22.0374527
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6991653, 13.7040024
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4221039, 12.4143028
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1206589, 16.1040726
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1683960, 12.1635284
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1357880, 14.1245346
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7938919, 12.7880287
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1798859, 14.1797447
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3451004, 18.3300171
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9092178, 12.9028168
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9771347, 15.9817810
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5447998, 14.5390511
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7466202, 17.7438736
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2754059, 15.2769165
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2446785, 11.2574844
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2206573, 16.2470474
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3706551, 11.3893623
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2536888, 11.2814407
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3181725, 12.3397827
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7721863, 16.7806244
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.3991432, 14.4073334
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2380981, 16.2398453
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5587006, 12.5813866
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9494324, 10.9592209
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5262794, 11.5285645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1742

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1208595, upper bound: 5.2061608
time: 35.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1146718, upper bound: 5.2123455
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4569626, 17.4855957
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8156967, 11.8401489
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9813690, 10.9974861
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9529572, 10.9618835
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1373291, 14.1596451
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1647263, 12.1705399
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5718651, 11.5441170
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9425659, 11.9702835
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6041679, 10.6423225
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4336014, 16.4501572
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0143700, 17.0534210
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0537033, 15.0761909
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4517097, 15.4499207
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3106918, 16.3149300
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7220154, 19.7882996
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1379318, 12.1495094
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2041016, 16.2227249
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0374603, 22.0860062
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.7025223, 13.6991615
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4126358, 12.4221039
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1042633, 16.1206589
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1623306, 12.1683960
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1243286, 14.1357841
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7876205, 12.7938919
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1781464, 14.1798897
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3300171, 18.3452988
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9026031, 12.9092178
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9807434, 15.9771347
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5379410, 14.5448036
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7430420, 17.7466202
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2745132, 15.2754021
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2567749, 11.2446785
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2463531, 16.2206497
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3893623, 11.3713608
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2814407, 11.2537575
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3397789, 12.3183060
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7779617, 16.7721863
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4073296, 14.3981400
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2402115, 16.2380905
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5809479, 12.5587044
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9586983, 10.9494324
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5285645, 11.5263786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1742

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2123455, upper bound: 5.1146718
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2061608, upper bound: 5.1208595
time: 15.57 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 25.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.01
Output dim: 35, lower bound: -5.1208595, upper bound: 5.2061608
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 25.01
Output dim: 35, lower bound: -5.1146718, upper bound: 5.2123455
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 25.01
Output dim: 35, lower bound: -5.2123455, upper bound: 5.1146718
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.01
Output dim: 35, lower bound: -5.2061608, upper bound: 5.1208595

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4649582, 17.4317589
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8164253, 11.7869453
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9804268, 10.9592628
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9409180, 10.9255180
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1336823, 14.1057053
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1519127, 12.1410446
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5301094, 11.5600929
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9423027, 11.9085770
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6106758, 10.5634632
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4471970, 16.4297714
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0527878, 17.0136566
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0825195, 15.0595741
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4292526, 15.4375114
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3061981, 16.2997322
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7792892, 19.7110405
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1350250, 12.1203613
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2213440, 16.2046623
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.1114120, 22.0596542
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6843376, 13.6908684
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4156723, 12.4083672
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1199875, 16.1034164
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1700211, 12.1650772
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1271591, 14.1169739
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7900085, 12.7846298
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1706161, 14.1716194
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3433609, 18.3284912
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9078522, 12.9015694
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9645538, 15.9707565
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5494614, 14.5433846
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7466660, 17.7439194
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2598190, 15.2631149
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2307777, 11.2455788
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2120056, 16.2400131
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3596725, 11.3797951
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2439194, 11.2725563
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3081436, 12.3306999
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7574310, 16.7672806
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.3917961, 14.4006882
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2448196, 16.2473602
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5521278, 12.5773926
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9413261, 10.9520721
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5209846, 11.5238380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.1031379, upper bound: 5.2118856
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1141144, upper bound: 5.1970737
time: 6.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4317551, 17.4637260
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.7869453, 11.8149796
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9592628, 10.9780769
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9255180, 10.9377861
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1057053, 14.1319275
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1410484, 12.1497765
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5596046, 11.5301094
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9085770, 11.9403992
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.5634613, 10.6064472
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4304352, 16.4471970
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0137558, 17.0527954
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0594025, 15.0825157
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4335861, 15.4292526
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.2997360, 16.3053131
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7110367, 19.7778435
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1203613, 12.1341858
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2026367, 16.2213478
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0596542, 22.1117935
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6893806, 13.6843414
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4066925, 12.4156761
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1036072, 16.1199875
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1638718, 12.1700172
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1167831, 14.1271629
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7842178, 12.7900124
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1700211, 14.1706161
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3284912, 18.3435745
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9013596, 12.9078522
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9697189, 15.9645538
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5422668, 14.5494652
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7430954, 17.7466660
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2607040, 15.2598267
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2448616, 11.2307777
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2393188, 16.2120056
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3797951, 11.3603821
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2725563, 11.2439919
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3307037, 12.3082733
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7646255, 16.7574310
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.4006920, 14.3907967
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2477341, 16.2448196
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5769463, 12.5521240
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9515495, 10.9413261
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5238380, 11.5210762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1970737, upper bound: 5.1141144
time: 12.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2118856, upper bound: 5.1031379
time: 22.58 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 37.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 37.78
Output dim: 35, lower bound: -5.1031379, upper bound: 5.2118856
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 37.78
Output dim: 35, lower bound: -5.1141144, upper bound: 5.1970737
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 37.78
Output dim: 35, lower bound: -5.1970737, upper bound: 5.1141144
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 37.78
Output dim: 35, lower bound: -5.2118856, upper bound: 5.1031379

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4587402, 17.4218292
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.8106499, 11.7777252
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9760742, 10.9523163
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9395561, 10.9233513
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.1286545, 14.0976715
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1514854, 12.1403542
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5156517, 11.5509224
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.9364243, 11.8991890
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.6038208, 10.5525169
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4437561, 16.4242706
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -17.0420151, 16.9970551
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0840530, 15.0580597
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4262466, 15.4353561
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.3044205, 16.2969170
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.7609863, 19.6818047
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1324310, 12.1162186
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.2160416, 16.1969376
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0991364, 22.0381622
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6835785, 13.6898308
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4157906, 12.4075584
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1219482, 16.1028137
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1703796, 12.1645851
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1253586, 14.1157646
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7899666, 12.7842636
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1703949, 14.1715660
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3460007, 18.3276291
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9079742, 12.9013252
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9644699, 15.9707336
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5497513, 14.5428734
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7465057, 17.7438736
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2598801, 15.2628860
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2257690, 11.2425423
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.1997147, 16.2323151
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3511963, 11.3741493
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2340965, 11.2664490
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.2996063, 12.3252487
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7527466, 16.7646255
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.3864326, 14.3963051
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2433777, 16.2463989
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5378609, 12.5684395
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9369049, 10.9493904
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5174332, 11.5215149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 722

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.0931761, upper bound: 5.2115236
time: 14.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1022925, upper bound: 5.1978229
time: 16.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.6479721, -32.6069870, -57.6479721, -32.6069870, -17.4218292, 17.4575081
1: -39.1900558, -20.1917152, -39.1900558, -20.1917152, -11.7777252, 11.8092003
2: -27.2321835, -11.1515217, -27.2321835, -11.1515217, -10.9523163, 10.9737282
3: -31.5792885, -14.0754833, -31.5792885, -14.0754833, -10.9233513, 10.9364281
4: -29.3921814, -8.6246052, -29.3921814, -8.6246052, -14.0976715, 14.1268959
5: -31.7481346, -13.5292301, -31.7481346, -13.5292301, -12.1403542, 12.1493568
6: -14.9053602, 2.8875151, -14.9053602, 2.8875151, -11.5504341, 11.5156517
7: -46.6423492, -25.5242538, -46.6423492, -25.5242538, -11.8991890, 11.9345207
8: -41.4315453, -19.8229351, -41.4315453, -19.8229351, -10.5525169, 10.5996017
9: -24.2807198, -5.1221490, -24.2807198, -5.1221490, -16.4249268, 16.4437561
10: -52.0965881, -29.6334991, -52.0965881, -29.6334991, -16.9971619, 17.0420227
11: -47.9131203, -27.0811577, -47.9131203, -27.0811577, -15.0578842, 15.0840454
12: -13.3641434, 5.9247103, -13.3641434, 5.9247103, -15.4314270, 15.4262505
13: -9.2810421, 9.7622128, -9.2810421, 9.7622128, -16.2969131, 16.3035545
14: -86.0954437, -59.4731178, -86.0954437, -59.4731178, -19.6818085, 19.7595329
15: -29.5717735, -11.9090519, -29.5717735, -11.9090519, -12.1162186, 12.1315918
16: -43.3784218, -22.5432358, -43.3784218, -22.5432358, -16.1949081, 16.2160416
17: -99.9770126, -70.0068512, -99.9770126, -70.0068512, -22.0381622, 22.0992203
18: -17.7611008, 3.4711514, -17.7611008, 3.4711514, -13.6883469, 13.6835823
19: -21.0215797, -6.4331346, -21.0215797, -6.4331346, -12.4058876, 12.4157944
20: -8.1969862, 5.5919609, -8.1969862, 5.5919609, -13.7889471, 13.7889471
21: -30.4836617, -12.1329889, -30.4836617, -12.1329889, -16.1029968, 16.1219482
22: -24.8078537, -8.3538189, -24.8078537, -8.3538189, -12.1633759, 12.1703758
23: -16.8797493, 0.1667442, -16.8797493, 0.1667442, -14.1155701, 14.1253548
24: -8.0182304, 6.9069571, -8.0182304, 6.9069571, -12.7838554, 12.7899666
25: -4.5950279, 11.7350121, -4.5950279, 11.7350121, -14.1699677, 14.1703987
26: -23.0610523, -1.5558355, -23.0610523, -1.5558355, -18.3276291, 18.3461227
27: -17.8111534, -3.7836523, -17.8111534, -3.7836523, -12.9011154, 12.9079742
28: -3.3356719, 16.1748238, -3.3356719, 16.1748238, -15.9696960, 15.9644699
29: -41.7373505, -23.3485413, -41.7373505, -23.3485413, -14.5417557, 14.5497551
30: -11.8002129, 7.2570019, -11.8002129, 7.2570019, -17.7430496, 17.7465057
31: -22.9123573, -4.3753605, -22.9123573, -4.3753605, -15.2604752, 15.2598724
32: -3.8060832, 10.6032219, -3.8060832, 10.6032219, -11.2418251, 11.2257690
33: 10.4932423, 30.8885670, 10.4932423, 30.8885670, -16.2316132, 16.1997147
34: 11.2562475, 28.9890003, 11.2562475, 28.9890003, -11.3741493, 11.3519020
35: 22.9045944, 40.4929657, 22.9045944, 40.4929657, -11.2664490, 11.2341728
36: 17.9324532, 34.5324326, 17.9324532, 34.5324326, -12.3252487, 12.2997322
37: 7.8623333, 28.1359882, 7.8623333, 28.1359882, -16.7619247, 16.7527466
38: 6.5726662, 26.5975685, 6.5726662, 26.5975685, -14.3963051, 14.3854294
39: 5.6902776, 25.9564323, 5.6902776, 25.9564323, -16.2466965, 16.2433777
40: 0.5834551, 19.8678665, 0.5834551, 19.8678665, -12.5677376, 12.5378647
41: -4.0929298, 9.0923929, -4.0929298, 9.0923929, -10.9488754, 10.9369049
42: -27.5886040, -10.8436394, -27.5886040, -10.8436394, -11.5215149, 11.5175247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=116, inp2_unstable=116, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1495
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1477

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 722

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.1978229, upper bound: 5.1022925
time: 13.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2115236, upper bound: 5.0931761
time: 7.71 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 23.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 23.18
Output dim: 35, lower bound: -5.0931761, upper bound: 5.2115236
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 23.18
Output dim: 35, lower bound: -5.1022925, upper bound: 5.1978229
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 23.18
Output dim: 35, lower bound: -5.1978229, upper bound: 5.1022925
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 23.18
Output dim: 35, lower bound: -5.2115236, upper bound: 5.0931761

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 21.56 + 695.67 = 717.23 seconds
