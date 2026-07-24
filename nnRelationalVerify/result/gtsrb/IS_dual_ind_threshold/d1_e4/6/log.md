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
execution time: IAR + RelationalAnalysis = 2.38 + 19.20 = 21.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 35, lower bound: -5.2169764, upper bound: 5.2169764

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 623

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 629

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2156483, upper bound: 5.1957970
time: 5.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2165735, upper bound: 5.2165731
time: 9.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.58
Output dim: 35, lower bound: -5.2156483, upper bound: 5.1957970
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.58
Output dim: 35, lower bound: -5.2165735, upper bound: 5.2165731

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -57.6297188, -32.6154099, -57.6383438, -32.6081009, -17.5531273, 17.5562363
1: -39.1787872, -20.1988029, -39.1839066, -20.1928139, -11.8920555, 11.8873672
2: -27.2243958, -11.1555328, -27.2280865, -11.1521091, -11.0201645, 11.0199013
3: -31.5740795, -14.0808353, -31.5767879, -14.0768585, -10.9574623, 10.9579086
4: -29.3627014, -8.6508217, -29.3753433, -8.6258345, -14.1806946, 14.1687317
5: -31.7452393, -13.5369415, -31.7474976, -13.5310965, -12.1702194, 12.1674538
6: -14.8889151, 2.8867273, -14.8958893, 2.8868790, -11.6374321, 11.6440964
7: -46.6321678, -25.5292988, -46.6369934, -25.5250092, -12.0367851, 12.0394020
8: -41.4221497, -19.8300552, -41.4262352, -19.8238869, -10.7238922, 10.7233238
9: -24.2518196, -5.1470876, -24.2639408, -5.1237144, -16.4653549, 16.4550018
10: -52.0489616, -29.6796894, -52.0685654, -29.6359749, -17.1294594, 17.1056671
11: -47.9043732, -27.0940018, -47.9108925, -27.0882130, -15.0511627, 15.0656166
12: -13.3269444, 5.8972006, -13.3436470, 5.9222479, -15.4184875, 15.4095116
13: -9.2757225, 9.7456179, -9.2796516, 9.7550917, -16.3178635, 16.2989426
14: -86.0260010, -59.5205421, -86.0563660, -59.4738541, -19.9247818, 19.9420395
15: -29.5510349, -11.9265804, -29.5601082, -11.9104633, -12.1473808, 12.1403694
16: -43.3584518, -22.5543098, -43.3675308, -22.5449772, -16.2443466, 16.2428284
17: -99.9401321, -70.0230560, -99.9572144, -70.0076218, -22.1427689, 22.1702652
18: -17.7319603, 3.4544697, -17.7454853, 3.4704142, -13.6772766, 13.6890907
19: -21.0079651, -6.4522963, -21.0200615, -6.4442911, -12.4040909, 12.4146500
20: -8.1837692, 5.5877128, -8.1920033, 5.5897264, -13.7734957, 13.7797165
21: -30.4588890, -12.1651154, -30.4807434, -12.1515322, -16.0863113, 16.1002731
22: -24.8010712, -8.3561039, -24.8055458, -8.3547497, -12.1542244, 12.1667633
23: -16.8499527, 0.1311326, -16.8780327, 0.1454526, -14.1164932, 14.1333008
24: -8.0112839, 6.8978643, -8.0159702, 6.9019442, -12.7582779, 12.7726364
25: -4.5658846, 11.7024870, -4.5933609, 11.7159519, -14.1403122, 14.1554680
26: -23.0321484, -1.5635445, -23.0475674, -1.5569057, -18.2899323, 18.3100815
27: -17.7988949, -3.7847426, -17.8055553, -3.7845640, -12.8907852, 12.9051514
28: -3.3154132, 16.1542492, -3.3339849, 16.1628208, -15.9491272, 15.9606857
29: -41.7260361, -23.3604660, -41.7356567, -23.3553772, -14.5347595, 14.5393219
30: -11.7781610, 7.2302732, -11.7980461, 7.2420340, -17.7102203, 17.7222443
31: -22.9031010, -4.3805952, -22.9101734, -4.3779993, -15.2478027, 15.2722092
32: -3.7666535, 10.5729609, -3.7826774, 10.6008186, -11.2457275, 11.2331276
33: 10.5216675, 30.8580437, 10.4963093, 30.8705940, -16.3066254, 16.3030853
34: 11.2701015, 28.9870415, 11.2631035, 28.9875889, -11.4337654, 11.4393005
35: 22.9435558, 40.4564857, 22.9068146, 40.4714584, -11.3178864, 11.3371239
36: 17.9384899, 34.5268784, 17.9336815, 34.5294266, -12.3929138, 12.3972092
37: 7.9065728, 28.0812035, 7.8653641, 28.1037197, -16.7417526, 16.7442818
38: 6.5895977, 26.5936699, 6.5797138, 26.5960827, -14.4182281, 14.4261398
39: 5.6963749, 25.9522629, 5.6926112, 25.9547958, -16.2295456, 16.2254944
40: 0.6130009, 19.8482151, 0.5996552, 19.8671951, -12.6116714, 12.6045532
41: -4.0885601, 9.0901670, -4.0912237, 9.0901775, -10.9663010, 10.9552650
42: -27.5826492, -10.8465137, -27.5863705, -10.8450651, -11.5237389, 11.5267410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=116, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 623

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 753

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2138652, upper bound: 5.1749026
time: 6.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2152091, upper bound: 5.1953583
time: 17.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -57.6468773, -32.6073914, -57.6473312, -32.6072006, -17.5672150, 17.5687790
1: -39.1889725, -20.1922359, -39.1893616, -20.1920261, -11.9022331, 11.9065628
2: -27.2318077, -11.1519060, -27.2319870, -11.1517458, -11.0243645, 11.0276222
3: -31.5767860, -14.0759993, -31.5778122, -14.0757980, -10.9594498, 10.9593811
4: -29.3908615, -8.6250525, -29.3914185, -8.6249094, -14.1835175, 14.2095718
5: -31.7477150, -13.5309076, -31.7478905, -13.5302153, -12.1736336, 12.1719818
6: -14.8954983, 2.8873191, -14.8992863, 2.8873916, -11.6406860, 11.6516609
7: -46.6418610, -25.5249290, -46.6419754, -25.5246696, -12.0434990, 12.0417671
8: -41.4311218, -19.8234558, -41.4313049, -19.8232384, -10.7249413, 10.7322674
9: -24.2790718, -5.1225166, -24.2797413, -5.1224012, -16.4928513, 16.4948883
10: -52.0945168, -29.6343861, -52.0953865, -29.6340389, -17.1269836, 17.1766052
11: -47.9127922, -27.0826988, -47.9129333, -27.0820656, -15.0894852, 15.0760384
12: -13.3620062, 5.9241924, -13.3628597, 5.9244070, -15.4192200, 15.4533310
13: -9.2801189, 9.7576675, -9.2804794, 9.7593117, -16.3102112, 16.3360138
14: -86.0916138, -59.4732590, -86.0931625, -59.4731903, -19.9869308, 19.9550362
15: -29.5703678, -11.9092560, -29.5709171, -11.9091759, -12.1613503, 12.1677399
16: -43.3771210, -22.5440903, -43.3776779, -22.5437565, -16.2586746, 16.2614784
17: -99.9744644, -70.0071259, -99.9753647, -70.0070572, -22.2169724, 22.1577911
18: -17.7589283, 3.4708362, -17.7597237, 3.4709599, -13.7164078, 13.6851883
19: -21.0212631, -6.4361987, -21.0213547, -6.4351130, -12.4375191, 12.4331665
20: -8.1942339, 5.5913219, -8.1952648, 5.5915556, -13.7857895, 13.7865868
21: -30.4831772, -12.1357574, -30.4833679, -12.1351509, -16.1371384, 16.1208572
22: -24.8067265, -8.3550739, -24.8071537, -8.3545856, -12.1798706, 12.1668434
23: -16.8794346, 0.1650360, -16.8795567, 0.1657455, -14.1702118, 14.1489754
24: -8.0180111, 6.9028654, -8.0180893, 6.9041462, -12.7950745, 12.7813072
25: -4.5946922, 11.7334633, -4.5948238, 11.7338581, -14.1901703, 14.1810875
26: -23.0585976, -1.5561376, -23.0595856, -1.5559998, -18.3387451, 18.3059845
27: -17.8051319, -3.7839055, -17.8073769, -3.7838025, -12.9121094, 12.8955879
28: -3.3350332, 16.1738167, -3.3352799, 16.1742439, -15.9827499, 15.9812851
29: -41.7365189, -23.3491459, -41.7368279, -23.3488998, -14.5501404, 14.5440712
30: -11.7998781, 7.2555227, -11.8000097, 7.2561293, -17.7526169, 17.7474365
31: -22.9120693, -4.3805828, -22.9121895, -4.3785400, -15.3032227, 15.2770042
32: -3.8030362, 10.6026573, -3.8036425, 10.6028643, -11.2524757, 11.2863808
33: 10.4939556, 30.8861580, 10.4936752, 30.8866158, -16.3139420, 16.3147659
34: 11.2654591, 28.9887199, 11.2616444, 28.9888420, -11.4388504, 11.4452667
35: 22.9053974, 40.4914322, 22.9050999, 40.4920692, -11.3693771, 11.3238182
36: 17.9339180, 34.5314865, 17.9333534, 34.5318604, -12.3988113, 12.4023972
37: 7.8634181, 28.1353569, 7.8630252, 28.1356087, -16.7764511, 16.7492714
38: 6.5817747, 26.5972137, 6.5780163, 26.5973587, -14.4283180, 14.4315872
39: 5.6910810, 25.9518948, 5.6907458, 25.9537373, -16.2317963, 16.2417145
40: 0.5863500, 19.8676834, 0.5852757, 19.8677635, -12.6081238, 12.6406937
41: -4.0891538, 9.0919609, -4.0906849, 9.0921459, -10.9693680, 10.9872818
42: -27.5848503, -10.8441095, -27.5864239, -10.8439341, -11.5325394, 11.5330772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=116, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 623

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 753

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2148212, upper bound: 5.1957276
time: 16.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2161341, upper bound: 5.2161335
time: 18.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 37.52 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 37.52
Output dim: 35, lower bound: -5.2138652, upper bound: 5.1749026
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 37.52
Output dim: 35, lower bound: -5.2152091, upper bound: 5.1953583
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 37.52
Output dim: 35, lower bound: -5.2148212, upper bound: 5.1957276
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 37.52
Output dim: 35, lower bound: -5.2161341, upper bound: 5.2161335

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -57.6239624, -32.6478882, -57.6139069, -32.6621094, -17.4929848, 17.4960022
1: -39.1767311, -20.2206917, -39.1692505, -20.2296829, -11.8529167, 11.8499908
2: -27.2221794, -11.1715736, -27.2181568, -11.1790171, -10.9904747, 10.9930458
3: -31.5701542, -14.0886679, -31.5684280, -14.0901136, -10.9369431, 10.9367714
4: -29.3600121, -8.6687851, -29.3621445, -8.6565189, -14.1468964, 14.1363525
5: -31.7437286, -13.5543098, -31.7400093, -13.5594606, -12.1403236, 12.1418457
6: -14.8659611, 2.8849239, -14.8566933, 2.8711853, -11.5950279, 11.6021538
7: -46.6304626, -25.5600967, -46.6175461, -25.5768147, -11.9836464, 11.9892387
8: -41.4210587, -19.8732433, -41.4084892, -19.8968849, -10.6500549, 10.6618843
9: -24.2499161, -5.1616969, -24.2537003, -5.1480017, -16.4351578, 16.4263992
10: -52.0483780, -29.7245522, -52.0456772, -29.7122650, -17.0488510, 17.0309296
11: -47.9022293, -27.1149502, -47.8947830, -27.1225471, -15.0230255, 15.0502129
12: -13.3172541, 5.8914223, -13.3267612, 5.9090123, -15.3862839, 15.3798828
13: -9.2556477, 9.7304192, -9.2419567, 9.7289228, -16.2701721, 16.2439651
14: -86.0202484, -59.5825195, -86.0053787, -59.5781479, -19.8148575, 19.8296928
15: -29.5465279, -11.9414787, -29.5496063, -11.9353523, -12.1169243, 12.1142616
16: -43.3544388, -22.5854111, -43.3500443, -22.5948410, -16.1876831, 16.1986847
17: -99.9353027, -70.0714111, -99.9214935, -70.0895386, -22.0577927, 22.1050034
18: -17.7264271, 3.4493473, -17.7374821, 3.4602699, -13.6590462, 13.6756821
19: -21.0015526, -6.4743571, -21.0008202, -6.4800320, -12.3692780, 12.3913193
20: -8.1768188, 5.5740871, -8.1789522, 5.5666437, -13.7434626, 13.7530394
21: -30.4540405, -12.1900473, -30.4584045, -12.1927309, -16.0458527, 16.0752792
22: -24.7942848, -8.3622570, -24.7916527, -8.3655844, -12.1314697, 12.1458054
23: -16.8454990, 0.1081336, -16.8633080, 0.1069617, -14.0764389, 14.1011276
24: -8.0037041, 6.8931246, -8.0035248, 6.8928809, -12.7382660, 12.7531967
25: -4.5575466, 11.6903896, -4.5754051, 11.6949272, -14.1125488, 14.1291733
26: -23.0232048, -1.5737712, -23.0316162, -1.5769873, -18.2499237, 18.2828293
27: -17.7925968, -3.7902560, -17.7941608, -3.7946229, -12.8693542, 12.8875198
28: -3.3052764, 16.1479321, -3.3162494, 16.1508942, -15.9253464, 15.9353790
29: -41.7225571, -23.3706284, -41.7262611, -23.3726139, -14.5137177, 14.5238266
30: -11.7740326, 7.2250662, -11.7901697, 7.2335482, -17.6889877, 17.7027969
31: -22.8912735, -4.4011393, -22.8927040, -4.4121680, -15.2059174, 15.2411537
32: -3.7470405, 10.5713387, -3.7491918, 10.5936146, -11.2170563, 11.1993256
33: 10.5681591, 30.8563461, 10.5743904, 30.8548717, -16.2440262, 16.2252274
34: 11.3088217, 28.9843693, 11.3280859, 28.9637508, -11.3715477, 11.3728485
35: 22.9933796, 40.4559402, 22.9901161, 40.4516525, -11.2488747, 11.2546043
36: 17.9899025, 34.5259552, 18.0202980, 34.5138321, -12.3283043, 12.3136406
37: 7.9160194, 28.0770187, 7.8801708, 28.0955772, -16.7179947, 16.7206345
38: 6.6355605, 26.5914001, 6.6571674, 26.5843925, -14.3566742, 14.3431587
39: 5.7369905, 25.9503326, 5.7612486, 25.9492455, -16.1783981, 16.1539078
40: 0.6318312, 19.8445930, 0.6314764, 19.8531647, -12.5716667, 12.5651703
41: -4.0753908, 9.0875378, -4.0685797, 9.0783873, -10.9405899, 10.9287872
42: -27.5790806, -10.8537655, -27.5804119, -10.8604650, -11.5057869, 11.5146599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1299

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2097428, upper bound: 5.1488400
time: 5.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2097428, upper bound: 5.1708030
time: 7.64 seconds

## BFS IS instance: IS_A1_B2

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

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 623

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2110528, upper bound: 5.1692878
time: 5.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2110528, upper bound: 5.1912179
time: 13.72 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -57.6410713, -32.6398849, -57.6228714, -32.6612244, -17.5070992, 17.5085793
1: -39.1869049, -20.2141533, -39.1747169, -20.2288666, -11.8631058, 11.8692055
2: -27.2295685, -11.1679430, -27.2220421, -11.1786804, -10.9946861, 11.0007668
3: -31.5728779, -14.0838280, -31.5694122, -14.0890675, -10.9389343, 10.9382477
4: -29.3881683, -8.6430073, -29.3781967, -8.6555967, -14.1497459, 14.1771965
5: -31.7462063, -13.5482569, -31.7404175, -13.5586023, -12.1437454, 12.1463585
6: -14.8725300, 2.8855472, -14.8600941, 2.8717103, -11.5982819, 11.6096687
7: -46.6400986, -25.5557671, -46.6225853, -25.5765018, -11.9903374, 11.9915924
8: -41.4300308, -19.8666401, -41.4135323, -19.8962135, -10.6510849, 10.6708431
9: -24.2771416, -5.1370955, -24.2695274, -5.1466861, -16.4626846, 16.4663162
10: -52.0939331, -29.6792641, -52.0724869, -29.7103729, -17.0463791, 17.1018524
11: -47.9106140, -27.1036148, -47.8967514, -27.1163998, -15.0613251, 15.0606003
12: -13.3523397, 5.9184465, -13.3459768, 5.9111481, -15.3870392, 15.4236603
13: -9.2600574, 9.7425003, -9.2427654, 9.7331276, -16.2625046, 16.2810783
14: -86.0858765, -59.5351982, -86.0421600, -59.5775108, -19.8770065, 19.8425980
15: -29.5658340, -11.9241619, -29.5604210, -11.9340639, -12.1309052, 12.1416206
16: -43.3730812, -22.5751915, -43.3601570, -22.5935841, -16.2020035, 16.2173080
17: -99.9696732, -70.0555344, -99.9397278, -70.0890198, -22.1320038, 22.0925140
18: -17.7533455, 3.4657526, -17.7517090, 3.4607875, -13.6982117, 13.6717377
19: -21.0148773, -6.4582305, -21.0021591, -6.4708018, -12.4026947, 12.4098396
20: -8.1872635, 5.5777264, -8.1821747, 5.5684910, -13.7557545, 13.7599010
21: -30.4783478, -12.1607008, -30.4609966, -12.1763458, -16.0967407, 16.0958633
22: -24.7999268, -8.3612080, -24.7932701, -8.3654108, -12.1571198, 12.1459045
23: -16.8749924, 0.1420264, -16.8648014, 0.1272304, -14.1301804, 14.1168251
24: -8.0104418, 6.8981509, -8.0056620, 6.8950872, -12.7750626, 12.7618599
25: -4.5863609, 11.7213945, -4.5768476, 11.7128029, -14.1623993, 14.1548157
26: -23.0496311, -1.5663862, -23.0436707, -1.5761194, -18.2987213, 18.2787094
27: -17.7988796, -3.7894459, -17.7959518, -3.7938678, -12.8906326, 12.8779259
28: -3.3248885, 16.1675339, -3.3175571, 16.1623478, -15.9589462, 15.9559631
29: -41.7330475, -23.3592949, -41.7274895, -23.3661461, -14.5290680, 14.5285683
30: -11.7957888, 7.2503223, -11.7921467, 7.2476196, -17.7313995, 17.7279739
31: -22.9002380, -4.4011135, -22.8947487, -4.4126811, -15.2613754, 15.2459145
32: -3.7834520, 10.6010122, -3.7701654, 10.5956354, -11.2237854, 11.2525749
33: 10.5404444, 30.8844299, 10.5717411, 30.8708992, -16.2513657, 16.2369003
34: 11.3041544, 28.9860477, 11.3266029, 28.9649696, -11.3766174, 11.3788185
35: 22.9552193, 40.4908752, 22.9884109, 40.4722557, -11.3003960, 11.2412529
36: 17.9853439, 34.5305557, 18.0199928, 34.5163155, -12.3342285, 12.3188553
37: 7.8728895, 28.1312180, 7.8778129, 28.1274891, -16.7527008, 16.7256088
38: 6.6276731, 26.5949116, 6.6554418, 26.5856609, -14.3667717, 14.3486214
39: 5.7317209, 25.9499550, 5.7593656, 25.9482212, -16.1806488, 16.1701050
40: 0.6051655, 19.8640671, 0.6170578, 19.8537006, -12.5681267, 12.6013184
41: -4.0759768, 9.0893574, -4.0680380, 9.0803165, -10.9436874, 10.9608192
42: -27.5812836, -10.8513546, -27.5804825, -10.8593292, -11.5145760, 11.5209846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=115, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1299

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2106641, upper bound: 5.1696537
time: 5.83 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2106641, upper bound: 5.1915589
time: 15.99 seconds

## BFS IS instance: IS_A2_B2

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

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 623

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2119453, upper bound: 5.1900564
time: 15.10 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 35, lower bound: -5.2119453, upper bound: 5.2119445
time: 8.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.50 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2097428, upper bound: 5.1488400
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2097428, upper bound: 5.1708030
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2110528, upper bound: 5.1692878
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2110528, upper bound: 5.1912179
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2106641, upper bound: 5.1696537
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2106641, upper bound: 5.1915589
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2119453, upper bound: 5.1900564
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 35, lower bound: -5.2119453, upper bound: 5.2119445

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -57.6306725, -32.6160698, -57.6415482, -32.6115036, -17.5190163, 17.5537567
1: -39.1756363, -20.2028427, -39.1846199, -20.1965141, -11.8611488, 11.8906631
2: -27.2305584, -11.1581821, -27.2312965, -11.1544933, -11.0076256, 11.0197830
3: -31.5733395, -14.0951233, -31.5753212, -14.0826998, -10.9491539, 10.9370613
4: -29.3711472, -8.6300697, -29.3844109, -8.6272783, -14.1412048, 14.1969261
5: -31.7456665, -13.5478821, -31.7470703, -13.5366631, -12.1635284, 12.1514473
6: -14.8940954, 2.8691916, -14.8979464, 2.8811302, -11.6279526, 11.5911369
7: -46.6390343, -25.5503578, -46.6408882, -25.5343437, -12.0025024, 12.0143738
8: -41.4305077, -19.8320484, -41.4308968, -19.8276711, -10.6785736, 10.7225914
9: -24.2610626, -5.1256056, -24.2735329, -5.1247187, -16.4625015, 16.4833069
10: -52.0690079, -29.6414490, -52.0867767, -29.6381855, -17.0367088, 17.1586990
11: -47.9081650, -27.0935097, -47.9110718, -27.0872173, -15.0733261, 15.0460739
12: -13.3067532, 5.9149170, -13.3437157, 5.9189448, -15.3642616, 15.4277496
13: -9.2648144, 9.7535801, -9.2741489, 9.7568798, -16.2861862, 16.3238297
14: -86.0432434, -59.4754524, -86.0760574, -59.4761047, -19.8760986, 19.9240379
15: -29.5346451, -11.9125614, -29.5583210, -11.9109907, -12.1156387, 12.1508141
16: -43.3693810, -22.5602646, -43.3747711, -22.5506172, -16.2248611, 16.2299309
17: -99.9189911, -70.0133514, -99.9558411, -70.0107574, -22.1269608, 22.1359634
18: -17.7374897, 3.4662597, -17.7522163, 3.4659600, -13.6977386, 13.6600075
19: -21.0113068, -6.4377675, -21.0176201, -6.4363041, -12.4197044, 12.4257507
20: -8.1839991, 5.5822010, -8.1912889, 5.5879178, -13.7719173, 13.7734900
21: -30.4729805, -12.1383791, -30.4794083, -12.1369352, -16.1137848, 16.1031036
22: -24.7663383, -8.3570452, -24.7930565, -8.3560505, -12.1379890, 12.1483917
23: -16.8743248, 0.1530225, -16.8775520, 0.1606832, -14.1399078, 14.1315346
24: -8.0146799, 6.8960514, -8.0165749, 6.9008818, -12.7859344, 12.7608719
25: -4.5857468, 11.7307329, -4.5913467, 11.7323570, -14.1726303, 14.1663094
26: -23.0073280, -1.5615377, -23.0412960, -1.5614884, -18.3015900, 18.2799301
27: -17.8013821, -3.7959595, -17.8058014, -3.7888522, -12.9074097, 12.8740387
28: -3.3279247, 16.1586838, -3.3321342, 16.1670532, -15.9681320, 15.9608688
29: -41.7136154, -23.3504906, -41.7287674, -23.3498077, -14.5319824, 14.5331230
30: -11.7933960, 7.2169242, -11.7962179, 7.2425056, -17.7378845, 17.7083588
31: -22.8984470, -4.3865004, -22.9069366, -4.3823447, -15.2827072, 15.2584877
32: -3.7944655, 10.5966806, -3.7999325, 10.6005726, -11.2297440, 11.2597809
33: 10.5010853, 30.8826904, 10.4980011, 30.8852043, -16.2910767, 16.2683792
34: 11.2693539, 28.9615593, 11.2642860, 28.9790573, -11.4258156, 11.3910522
35: 22.9116592, 40.4744644, 22.9090500, 40.4861832, -11.3602638, 11.2745018
36: 17.9481888, 34.5200386, 17.9400749, 34.5277328, -12.3787842, 12.3573341
37: 7.8732100, 28.1303539, 7.8669786, 28.1304111, -16.7445145, 16.7279510
38: 6.5897946, 26.5918350, 6.5823689, 26.5952492, -14.4164734, 14.4067879
39: 5.7060537, 25.9463158, 5.6974516, 25.9488678, -16.2017059, 16.2211304
40: 0.5889502, 19.8530540, 0.5869160, 19.8619003, -12.5873413, 12.5992622
41: -4.0875692, 9.0778942, -4.0896559, 9.0868950, -10.9473114, 10.9592361
42: -27.5834637, -10.8563795, -27.5854340, -10.8498354, -11.5099907, 11.5174179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 623

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 737

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2112353, upper bound: 5.1659951
time: 6.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2116501, upper bound: 5.1897613
time: 5.92 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -57.6531944, -32.5811234, -57.6437912, -32.6111946, -17.5428085, 17.5501442
1: -39.1870079, -20.1496124, -39.1866455, -20.1959572, -11.8691940, 11.9451065
2: -27.2392540, -11.1555157, -27.2309418, -11.1553621, -11.0142746, 11.0204010
3: -31.6208611, -14.0901423, -31.5755062, -14.0891571, -11.0049171, 10.9503403
4: -29.3779259, -8.5541382, -29.3806534, -8.6271343, -14.1497879, 14.2861137
5: -31.8006077, -13.5421553, -31.7468605, -13.5436163, -12.2295227, 12.1662331
6: -14.9396029, 2.8653088, -14.8975220, 2.8733215, -11.6987152, 11.5974121
7: -46.6768951, -25.5493965, -46.6408882, -25.5407963, -12.0468292, 12.0245895
8: -41.4261398, -19.8250542, -41.4275818, -19.8274803, -10.6753502, 10.7453461
9: -24.2828960, -5.0622845, -24.2767944, -5.1253896, -16.4745789, 16.5398026
10: -52.0916748, -29.5455837, -52.0927353, -29.6385918, -17.0513992, 17.2540894
11: -47.9471130, -27.1200981, -47.9116516, -27.1091175, -15.1601410, 15.0528145
12: -13.3542347, 6.0941420, -13.3518181, 5.9195089, -15.4007759, 15.6074448
13: -9.2784128, 9.7747688, -9.2696991, 9.7562714, -16.3084335, 16.3573418
14: -86.0937805, -59.3521309, -86.0827942, -59.4766045, -19.9891586, 19.9027061
15: -29.5579052, -11.8272047, -29.5572376, -11.9106998, -12.1399269, 12.2464867
16: -43.3837166, -22.5885525, -43.3752022, -22.5749187, -16.2878876, 16.2517014
17: -99.9687653, -69.8270187, -99.9677277, -70.0122681, -22.1609116, 22.2769318
18: -17.7487621, 3.5356832, -17.7517090, 3.4658318, -13.7710953, 13.6815605
19: -21.0240498, -6.4228554, -21.0186386, -6.4362855, -12.4339371, 12.4222679
20: -8.2128468, 5.5834761, -8.1916504, 5.5854349, -13.7982817, 13.7751265
21: -30.4888096, -12.1270323, -30.4807453, -12.1370182, -16.1604843, 16.0909271
22: -24.7936058, -8.2386351, -24.7919922, -8.3562269, -12.1618233, 12.2686806
23: -16.9140167, 0.1566423, -16.8777599, 0.1583145, -14.1905746, 14.1309967
24: -8.0250845, 6.8984518, -8.0160389, 6.8965163, -12.8360825, 12.7656631
25: -4.5914221, 11.7516823, -4.5903468, 11.7323437, -14.2119904, 14.1665726
26: -23.0423012, -1.4651191, -23.0412445, -1.5622623, -18.3328171, 18.3739700
27: -17.8370667, -3.7868302, -17.8051205, -3.7884083, -12.9683456, 12.8786201
28: -3.3769650, 16.1667633, -3.3321104, 16.1672459, -16.0208282, 15.9620972
29: -41.7368507, -23.2957134, -41.7324142, -23.3498592, -14.5551682, 14.5519524
30: -11.9336100, 7.2510996, -11.7969666, 7.2498069, -17.8752060, 17.7340240
31: -22.9113579, -4.3581581, -22.9082336, -4.3822484, -15.3065033, 15.2590408
32: -3.8056788, 10.6209106, -3.8011189, 10.6015558, -11.2308311, 11.3217239
33: 10.4755230, 30.8914108, 10.4980984, 30.8847084, -16.2810898, 16.3397293
34: 11.1939249, 28.9922409, 11.2644863, 28.9847221, -11.5029831, 11.4175682
35: 22.8351212, 40.4884338, 22.9095459, 40.4903183, -11.4279060, 11.2805252
36: 17.9070721, 34.5324821, 17.9408417, 34.5303612, -12.4055595, 12.3666573
37: 7.8500204, 28.1284504, 7.8679585, 28.1252422, -16.7353973, 16.7969513
38: 6.5938983, 26.6164207, 6.5871401, 26.5957184, -14.4118309, 14.4263420
39: 5.6978807, 25.9914570, 5.7026944, 25.9488640, -16.1887054, 16.3310547
40: 0.5089760, 19.8620110, 0.5873809, 19.8603172, -12.5773544, 12.6659698
41: -4.1099629, 9.0875025, -4.0889511, 9.0853662, -10.9364662, 11.0328102
42: -27.5893764, -10.8522177, -27.5854473, -10.8547297, -11.4838104, 11.5830421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=114, inp2_unstable=115, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=125, inp2_unstable=125, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=10, inp2_unstable=10, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=12, inp2_unstable=12, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 623

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 737

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2112353, upper bound: 5.1878979
time: 5.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 35, lower bound: -5.2116501, upper bound: 5.2116491
time: 6.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.81 seconds
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.81
Output dim: 35, lower bound: -5.2112353, upper bound: 5.1659951
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 13.81
Output dim: 35, lower bound: -5.2116501, upper bound: 5.1897613
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 13.81
Output dim: 35, lower bound: -5.2112353, upper bound: 5.1878979
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 13.81
Output dim: 35, lower bound: -5.2116501, upper bound: 5.2116491

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 21.58 + 191.39 = 212.97 seconds
