## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 1800 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444)
1: (-15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248)
2: (-12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801)
3: (-8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943)
4: (-12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427)
5: (-9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509)
6: (-27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0119476, 19.0119514)
7: (-13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115)
8: (-16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7938156, 31.7938194)
9: (-12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4043579, 21.4043617)
10: (-13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6689148, 34.6689186)
11: (-22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8888054, 33.8888054)
12: (-20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1794128, 36.1794128)
13: (-21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8892784, 25.8892822)
14: (-43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3938179, 34.3938103)
15: (-15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4708977, 24.4708939)
16: (-21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3633194, 33.3633156)
17: (-33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4835052, 52.4835129)
18: (-17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4217415, 24.4217415)
19: (-20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5478134, 21.5478134)
20: (-10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7429657, 19.7429657)
21: (-20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665)
22: (-22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3948288, 31.3948288)
23: (-19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3794556, 22.3794518)
24: (-26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5211906, 21.5211945)
25: (-13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3763962, 21.3764000)
26: (-28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7256012, 37.7256012)
27: (-28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5362625, 24.5362701)
28: (-18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0137711, 24.0137749)
29: (-32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8249359, 35.8249359)
30: (-18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7941742, 25.7941742)
31: (-18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1398239, 25.1398239)
32: (-21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3143997, 22.3144035)
33: (-39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2729187, 32.2729187)
34: (-30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8428612, 27.8428650)
35: (-30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2664719, 26.2664680)
36: (-31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5797386, 24.5797386)
37: (-47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6503754, 32.6503754)
38: (-40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7989540, 27.7989521)
39: (-50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4803543, 34.4803543)
40: (-41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8075867, 31.8075829)
41: (-31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0617027, 20.0617027)
42: (-18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5602455, 19.5602474)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.74 + 29.39 = 32.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.9025189, upper bound: 17.9025189

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1641

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8879850, upper bound: 17.9013316
time: 30.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9013316, upper bound: 17.8879850
time: 20.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 51.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 51.35
Output dim: 10, lower bound: -17.8879850, upper bound: 17.9013316
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 51.35
Output dim: 10, lower bound: -17.9013316, upper bound: 17.8879850

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0225105, 19.0229836
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7932205, 31.7930946
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3766708, 21.3807373
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6568832, 34.6569290
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8837433, 33.8830948
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1816788, 36.1835556
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8665314, 25.8686295
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3909760, 34.3908424
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4748688, 24.4744492
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3661118, 33.3710899
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4744339, 52.4733658
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4184761, 24.4170914
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5356102, 21.5336151
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7305222, 19.7292633
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3745422, 31.3704910
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3836517, 22.3835869
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5011292, 21.4974747
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3701973, 21.3698082
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7163467, 37.7149811
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5146942, 24.5115547
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0027313, 24.0005264
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8092422, 35.8061142
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7803154, 25.7775497
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1233215, 25.1204224
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3231010, 22.3239975
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2540512, 32.2571411
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8404160, 27.8405533
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2626343, 26.2634659
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5783348, 24.5786552
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6157761, 32.6210556
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7977219, 27.7978668
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4634018, 34.4662247
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7826271, 31.7860985
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0623589, 20.0645313
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5844173, 19.5910225

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8533434, upper bound: 17.9003891
time: 16.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8870448, upper bound: 17.8666718
time: 20.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0229836, 19.0225105
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7930984, 31.7932167
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3807373, 21.3766727
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6569290, 34.6568832
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8830948, 33.8837433
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1835556, 36.1816788
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8686218, 25.8665314
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3908463, 34.3909760
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4744492, 24.4748688
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3710861, 33.3661118
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4733658, 52.4744263
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4170914, 24.4184761
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5336189, 21.5356064
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7292633, 19.7305183
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3704910, 31.3745422
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3835907, 22.3836555
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4974747, 21.5011292
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3698082, 21.3701897
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7149811, 37.7163467
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5115509, 24.5146980
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0005264, 24.0027313
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8061142, 35.8092422
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7775536, 25.7803154
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1204147, 25.1233215
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3240013, 22.3231010
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2571411, 32.2540550
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8405533, 27.8404160
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2634659, 26.2626343
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5786552, 24.5783348
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6210556, 32.6157761
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7978668, 27.7977219
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4662247, 34.4634018
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7860985, 31.7826271
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0645294, 20.0623589
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5910206, 19.5844173

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8666718, upper bound: 17.8870448
time: 20.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9003891, upper bound: 17.8533434
time: 18.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 40.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 10, lower bound: -17.8533434, upper bound: 17.9003891
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 10, lower bound: -17.8870448, upper bound: 17.8666718
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 10, lower bound: -17.8666718, upper bound: 17.8870448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 10, lower bound: -17.9003891, upper bound: 17.8533434

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9823189, 18.9752750
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7934952, 31.7933998
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3241348, 21.3368530
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6175003, 34.6240311
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8755875, 33.8739243
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2058563, 36.2125320
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8181000, 25.8284111
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3319435, 34.3411522
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4681168, 24.4683685
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3441925, 33.3527832
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4324646, 52.4385681
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3801918, 24.3715630
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5155525, 21.5096054
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7148209, 19.7104683
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3703613, 31.3655853
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3771706, 22.3764610
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4631958, 21.4520683
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3675613, 21.3672333
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6955338, 37.6897736
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4709473, 24.4591751
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9909058, 23.9863701
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8053513, 35.8016205
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7690125, 25.7650375
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1054611, 25.0990448
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3212509, 22.3218765
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2370377, 32.2371521
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8167648, 27.8107567
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2465515, 26.2440338
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5486908, 24.5433159
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5838242, 32.5849838
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7408447, 27.7292175
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4428711, 34.4415855
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7560387, 31.7548828
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0215969, 20.0163307
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5892467, 19.5965633

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1749

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8998561
time: 25.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8748835
time: 25.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9748001, 18.9827919
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7935257, 31.7933693
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3327866, 21.3281994
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6239853, 34.6175461
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8745728, 33.8749428
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2106552, 36.2077332
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8263092, 25.8201904
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3412819, 34.3318100
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4687881, 24.4676971
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3478012, 33.3491707
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4396210, 52.4314117
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3729477, 24.3788071
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5116005, 21.5135574
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7117233, 19.7135658
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3696289, 31.3663177
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3765297, 22.3771057
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4557190, 21.4595451
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3676147, 21.3671837
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6911469, 37.6941605
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4623184, 24.4678040
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9885712, 23.9886971
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8047485, 35.8022156
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7678070, 25.7662506
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1019440, 25.1025658
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3209763, 22.3221436
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2340622, 32.2401237
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8106155, 27.8168983
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2432022, 26.2473831
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5429916, 24.5490150
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5797043, 32.5891037
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7290726, 27.7409859
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4387665, 34.4456940
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7514153, 31.7595062
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0141582, 20.0237656
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5899601, 19.5958519

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8661380
time: 30.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8865117, upper bound: 17.8411438
time: 24.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9827919, 18.9748001
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7933731, 31.7935219
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3282013, 21.3327885
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6175461, 34.6239853
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8749390, 33.8745728
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2077332, 36.2106552
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8201904, 25.8263168
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3318138, 34.3412857
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4677048, 24.4687805
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3491745, 33.3478012
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4314117, 52.4396286
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3788071, 24.3729477
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5135536, 21.5115967
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7135696, 19.7117233
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3663177, 31.3696289
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3771095, 22.3765259
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4595490, 21.4557228
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3671799, 21.3676147
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6941605, 37.6911469
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4678040, 24.4623184
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9886932, 23.9885788
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8022156, 35.8047485
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7662506, 25.7678032
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1025620, 25.1019440
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3221512, 22.3209801
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2401199, 32.2340660
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8169022, 27.8106194
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2473831, 26.2432022
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5490112, 24.5429916
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5891037, 32.5797005
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7409821, 27.7290726
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4456940, 34.4387627
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7595100, 31.7514153
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0237637, 20.0141563
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5958538, 19.5899601

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8865117
time: 31.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8615749
time: 23.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9752769, 18.9823189
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7934036, 31.7934914
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3368530, 21.3241348
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6240311, 34.6175003
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8739243, 33.8755875
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2125320, 36.2058563
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8284149, 25.8180962
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3411522, 34.3319435
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4683609, 24.4681129
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3527832, 33.3441925
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4385529, 52.4324646
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3715630, 24.3801918
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5096016, 21.5155487
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7104721, 19.7148170
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3655853, 31.3703613
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3764610, 22.3771706
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4520721, 21.4631996
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3672333, 21.3675652
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6897736, 37.6955338
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4591751, 24.4709473
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9863739, 23.9909019
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8016205, 35.8053513
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7650299, 25.7690163
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0990372, 25.1054668
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3218765, 22.3212471
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2371521, 32.2370377
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8107529, 27.8167610
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2440338, 26.2465515
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5433121, 24.5486946
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5849838, 32.5838242
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7292175, 27.7408409
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4415894, 34.4428711
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7548866, 31.7560387
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0163326, 20.0215931
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5965633, 19.5892467

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8528097
time: 25.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8998561, upper bound: 17.8278349
time: 20.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 48.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8998561
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8748835
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8661380
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8865117, upper bound: 17.8411438
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8865117
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8615749
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8528097
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.64
Output dim: 10, lower bound: -17.8998561, upper bound: 17.8278349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9890842, 18.9809132
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7944946, 31.7935257
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2888069, 21.3077354
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5913391, 34.6020508
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8701591, 33.8675156
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1869240, 36.1984825
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7778473, 25.7963638
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3211823, 34.3352356
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4681931, 24.4683723
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3374405, 33.3483620
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4546967, 52.4670105
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3676300, 24.3530235
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5006638, 21.4915161
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7033539, 19.6966934
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3648682, 31.3592224
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3724709, 22.3708572
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4373398, 21.4210892
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3687973, 21.3683357
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6825409, 37.6733780
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4362717, 24.4176292
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9825592, 23.9763832
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8032837, 35.7992401
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7612305, 25.7558517
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0958595, 25.0865440
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3237762, 22.3246651
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2419968, 32.2421341
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8350410, 27.8258286
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2512741, 26.2488060
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5432281, 24.5366745
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5918770, 32.5918274
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7505951, 27.7338371
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4472122, 34.4460526
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7736588, 31.7711067
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0170822, 20.0097656
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5983582, 19.6073761

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8262730, upper bound: 17.8996847
time: 18.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8276682, upper bound: 17.8984279
time: 17.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9879551, 18.9820404
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7936172, 31.7944031
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2950211, 21.3015251
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5955124, 34.5978699
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8691750, 33.8684959
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1918068, 36.1935997
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7860489, 25.7881622
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3260345, 34.3303909
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4681168, 24.4684410
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3397751, 33.3460274
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4609070, 52.4607925
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3616524, 24.3589993
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4974594, 21.4947128
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7010422, 19.6990051
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3640060, 31.3600845
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3715706, 22.3717613
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4322205, 21.4262085
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3686676, 21.3684616
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6791306, 37.6767807
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4294052, 24.4244957
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9809113, 23.9780312
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8029709, 35.7995529
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7598343, 25.7572556
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0929756, 25.0894279
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3240356, 22.3244057
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2420197, 32.2421150
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8318291, 27.8290405
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2513199, 26.2487602
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5420532, 24.5378494
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5906639, 32.5930328
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7454605, 27.7389736
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4473419, 34.4459305
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7722626, 31.7725029
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0150299, 20.0118179
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6000595, 19.6056747

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8512631, upper bound: 17.8747180
time: 23.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8526391, upper bound: 17.8734570
time: 20.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9815693, 18.9884300
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7945251, 31.7934952
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2974625, 21.2990837
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5978241, 34.5955658
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8691444, 33.8685303
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1917305, 36.1936874
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7860718, 25.7881393
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3305206, 34.3258934
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4688492, 24.4677010
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3410492, 33.3447533
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4618530, 52.4598465
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3603859, 24.3602676
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4967117, 21.4954681
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7002563, 19.6997910
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3641281, 31.3599548
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3718300, 22.3715019
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4298630, 21.4285660
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3688431, 21.3682861
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6781540, 37.6777573
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4276352, 24.4262581
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9802399, 23.9787025
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8026810, 35.7998428
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7600174, 25.7570648
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0923347, 25.0900650
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3235168, 22.3249321
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2390289, 32.2451019
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8289070, 27.8319702
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2479248, 26.2521553
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5375214, 24.5423737
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5877495, 32.5959473
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7388306, 27.7456036
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4431076, 34.4501648
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7690353, 31.7757263
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0096436, 20.0172024
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5990677, 19.6066628

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8600305, upper bound: 17.8659683
time: 19.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8614043, upper bound: 17.8647061
time: 19.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9804401, 18.9895592
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7936478, 31.7943726
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3036728, 21.2928715
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6020050, 34.5913849
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8681602, 33.8695145
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1966133, 36.1888046
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7942657, 25.7799416
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3353729, 34.3210487
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4687881, 24.4677696
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3433838, 33.3424149
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4680634, 52.4536285
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3544121, 24.3662434
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4935074, 21.4986687
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6979446, 19.7021027
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3632660, 31.3608246
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3709221, 22.3724060
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4247437, 21.4336891
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3687210, 21.3684120
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6747437, 37.6811676
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4207687, 24.4331245
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9785919, 23.9803581
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8023682, 35.8001556
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7586212, 25.7584686
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0894508, 25.0929508
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3237610, 22.3246727
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2390442, 32.2450867
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8256950, 27.8351784
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2479706, 26.2521095
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5363464, 24.5435524
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5865440, 32.5971565
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7336960, 27.7507401
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4432297, 34.4500389
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7676392, 31.7771263
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0075912, 20.0192528
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6007729, 19.6049614

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8849826, upper bound: 17.8409809
time: 32.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8863416, upper bound: 17.8397133
time: 20.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9895573, 18.9804382
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7943726, 31.7936478
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2928696, 21.3036728
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5913849, 34.6020050
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8695107, 33.8681602
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1888008, 36.1966095
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7799377, 25.7942657
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3210449, 34.3353691
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4677658, 24.4687843
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3424149, 33.3433838
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4536285, 52.4680634
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3662453, 24.3544083
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4986649, 21.4935074
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7021027, 19.6979446
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3608170, 31.3632660
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3724098, 22.3709221
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4336853, 21.4247437
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3684158, 21.3687172
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6811676, 37.6747437
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4331284, 24.4207726
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9803619, 23.9785881
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8001556, 35.8023682
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7584686, 25.7586174
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0929604, 25.0894432
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3246765, 22.3237686
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2450867, 32.2390442
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8351784, 27.8256912
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2521133, 26.2479706
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5435486, 24.5363503
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5971565, 32.5865479
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7507401, 27.7336922
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4500351, 34.4432335
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7771301, 31.7676353
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0192566, 20.0075912
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6049614, 19.6007729

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8397133, upper bound: 17.8863416
time: 19.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8849826
time: 20.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9884281, 18.9815674
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7934952, 31.7945251
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2990837, 21.2974606
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5955582, 34.5978241
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8685341, 33.8691406
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1936836, 36.1917267
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7881393, 25.7860680
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3258972, 34.3305244
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4677048, 24.4688568
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3447495, 33.3410492
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4598541, 52.4618530
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3602715, 24.3603840
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4954681, 21.4967079
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6997910, 19.7002563
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3599548, 31.3641281
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3715019, 22.3718262
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4285660, 21.4298630
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3682861, 21.3688469
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6777573, 37.6781540
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4262619, 24.4276390
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9786987, 23.9802399
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7998428, 35.8026886
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7570648, 25.7600212
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0900612, 25.0923290
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3249359, 22.3235092
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2451019, 32.2390289
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8319664, 27.8289032
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2521515, 26.2479248
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5423737, 24.5375290
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5959511, 32.5877533
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7456055, 27.7388287
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4501648, 34.4431076
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7757263, 31.7690353
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0172043, 20.0096436
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6066628, 19.5990696

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8647061, upper bound: 17.8614043
time: 20.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8659683, upper bound: 17.8600305
time: 18.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9820423, 18.9879570
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7944031, 31.7936172
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3015251, 21.2950211
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5978699, 34.5955200
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8684959, 33.8691788
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1935997, 36.1918106
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7881622, 25.7860489
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3303986, 34.3260269
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4684372, 24.4681206
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3460312, 33.3397751
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4608002, 52.4609070
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3590012, 24.3616524
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4947128, 21.4974632
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6990051, 19.7010422
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3600845, 31.3640060
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3717613, 22.3715668
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4262085, 21.4322205
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3684616, 21.3686714
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6767807, 37.6791306
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4244919, 24.4294014
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9780273, 23.9809113
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7995529, 35.8029709
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7572556, 25.7598305
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0894356, 25.0929661
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3244019, 22.3240356
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2421112, 32.2420158
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8290367, 27.8318329
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2487564, 26.2513199
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5378418, 24.5420532
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5930367, 32.5906677
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7389755, 27.7454586
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4459305, 34.4473419
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7725067, 31.7722588
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0118179, 20.0150299
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6056747, 19.6000595

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8734570, upper bound: 17.8526391
time: 18.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8747180, upper bound: 17.8512631
time: 19.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9809132, 18.9890842
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7935257, 31.7944946
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3077354, 21.2888069
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6020508, 34.5913391
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8675117, 33.8701630
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1984825, 36.1869278
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7963638, 25.7778473
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3352356, 34.3211823
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4683762, 24.4681892
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3483658, 33.3374405
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4670105, 52.4546967
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3530235, 24.3676281
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4915161, 21.5006638
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6966934, 19.7033539
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3592224, 31.3648682
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3708534, 22.3724709
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4210892, 21.4373398
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3683395, 21.3687935
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6733704, 37.6825333
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4176254, 24.4362679
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9763794, 23.9825630
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7992401, 35.8032837
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7558517, 25.7612343
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0865517, 25.0958500
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3246613, 22.3237762
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2421341, 32.2419968
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8258324, 27.8350449
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2488022, 26.2512741
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5366669, 24.5432320
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5918236, 32.5918732
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7338333, 27.7505951
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4460526, 34.4472160
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7711029, 31.7736588
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0097656, 20.0170803
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6073761, 19.5983582

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8984279, upper bound: 17.8276682
time: 20.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8262730
time: 32.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 54.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8262730, upper bound: 17.8996847
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8276682, upper bound: 17.8984279
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8512631, upper bound: 17.8747180
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8526391, upper bound: 17.8734570
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8600305, upper bound: 17.8659683
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8614043, upper bound: 17.8647061
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8849826, upper bound: 17.8409809
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8863416, upper bound: 17.8397133
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8397133, upper bound: 17.8863416
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8849826
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8647061, upper bound: 17.8614043
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8659683, upper bound: 17.8600305
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8734570, upper bound: 17.8526391
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8747180, upper bound: 17.8512631
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8984279, upper bound: 17.8276682
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.91
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8262730

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0004692, 18.9952469
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8006439, 31.7982979
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2617188, 21.2854462
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5878067, 34.5985069
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8603897, 33.8558083
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2056198, 36.2208519
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7448502, 25.7672386
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3113480, 34.3246346
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4813232, 24.4794235
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3201675, 33.3357964
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4501877, 52.4619751
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3527946, 24.3360901
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4831085, 21.4706650
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6912613, 19.6838417
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3439178, 31.3340912
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3674393, 22.3653793
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4085808, 21.3865776
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3566551, 21.3548470
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6628876, 37.6501007
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4063950, 24.3840179
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9696960, 23.9609413
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7856827, 35.7781143
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7461166, 25.7377167
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0752029, 25.0616646
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3438187, 22.3477440
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2216873, 32.2256050
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8313789, 27.8230095
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2467651, 26.2455406
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5394211, 24.5343742
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5762863, 32.5794296
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7447662, 27.7288780
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4296036, 34.4313736
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7512856, 31.7519913
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0244675, 20.0197353
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6316242, 19.6472759

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8123254, upper bound: 17.8988680
time: 23.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8256596, upper bound: 17.8789810
time: 27.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0034180, 18.9922981
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7992706, 31.7996788
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2665176, 21.2806473
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5877914, 34.5985184
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8584595, 33.8577385
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2092972, 36.2171783
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7487259, 25.7633629
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3105774, 34.3253975
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4792404, 24.4815102
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3248672, 33.3310928
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4496689, 52.4625092
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3506966, 24.3383465
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4798126, 21.4740219
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6904984, 19.6846466
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3397369, 31.3382721
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3669968, 22.3658295
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4028282, 21.3923225
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3553047, 21.3562012
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6592636, 37.6536713
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4026566, 24.3875580
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9671249, 23.9635124
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7821655, 35.7816391
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7430954, 25.7407341
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0709763, 25.0658951
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3468552, 22.3447075
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2254715, 32.2218246
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8322258, 27.8221588
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2480164, 26.2442856
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5409317, 24.5328712
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5794754, 32.5762405
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7456360, 27.7280102
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4325333, 34.4284401
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7545433, 31.7487297
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0270500, 20.0171528
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6385441, 19.6406403

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8976116
time: 24.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8270518, upper bound: 17.8777205
time: 30.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9993401, 18.9963760
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7997742, 31.7991753
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2679291, 21.2792320
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5919800, 34.5943298
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8594055, 33.8567924
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2105026, 36.2159691
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7530441, 25.7590370
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3161926, 34.3197899
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4812622, 24.4794922
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3225021, 33.3334618
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4564133, 52.4557648
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3468170, 24.3420658
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4799194, 21.4738617
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6889420, 19.6861534
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3430557, 31.3349533
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3665390, 22.3662834
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4034538, 21.3917007
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3565331, 21.3549728
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6594772, 37.6535110
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3995285, 24.3908806
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9680405, 23.9625931
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7853699, 35.7784348
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7447128, 25.7391167
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0723190, 25.0645504
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3440781, 22.3474884
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2217102, 32.2255859
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8281593, 27.8262215
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2467957, 26.2454948
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5382462, 24.5355492
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5750809, 32.5806351
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7396317, 27.7340126
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4297256, 34.4312515
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7498817, 31.7533913
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0224152, 20.0217876
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6333256, 19.6455727

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8360916, upper bound: 17.8739071
time: 32.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8506593, upper bound: 17.8581190
time: 20.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0022888, 18.9934273
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7983856, 31.8005562
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2727318, 21.2744331
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5919724, 34.5943375
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8574753, 33.8587227
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2141800, 36.2122955
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7569199, 25.7551651
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3154297, 34.3205528
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4791718, 24.4815788
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3272095, 33.3287582
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4558792, 52.4562912
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3447189, 24.3443222
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4766083, 21.4772224
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6881943, 19.6869583
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3388672, 31.3391342
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3660889, 22.3667336
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3977089, 21.3974457
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3551750, 21.3563232
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6558609, 37.6570740
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3957901, 24.3944206
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9654694, 23.9651642
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7818451, 35.7819519
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7416916, 25.7421379
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0680923, 25.0687809
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3471146, 22.3444519
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2254868, 32.2218056
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8290138, 27.8253708
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2480621, 26.2442436
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5397568, 24.5340462
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5782700, 32.5774460
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7405014, 27.7331467
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4326630, 34.4283142
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7531471, 31.7501259
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0249977, 20.0192051
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6402454, 19.6389389

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8726459
time: 19.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8520339, upper bound: 17.8568537
time: 20.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9929543, 19.0027637
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8006744, 31.7982674
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2703705, 21.2767944
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5942917, 34.5920181
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8593674, 33.8568268
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2104187, 36.2160530
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7530746, 25.7590179
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3206863, 34.3152924
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4819946, 24.4787521
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3237762, 33.3321877
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4573593, 52.4548187
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3455505, 24.3433342
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4791565, 21.4746170
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6881638, 19.6869392
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3431854, 31.3348236
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3667984, 22.3660240
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4011040, 21.3940544
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3567085, 21.3547974
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6585007, 37.6544876
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3977585, 24.3926430
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9673691, 23.9632645
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7850876, 35.7787170
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7449036, 25.7389259
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0716782, 25.0651875
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3435516, 22.3480110
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2187195, 32.2285767
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8252373, 27.8291473
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2434082, 26.2488899
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5337296, 24.5400734
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5721664, 32.5835495
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7330017, 27.7406425
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4254913, 34.4354858
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7466621, 31.7566147
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0170326, 20.0271721
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6323338, 19.6465645

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8434273, upper bound: 17.8653452
time: 21.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8592223, upper bound: 17.8508013
time: 26.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9958992, 18.9998169
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7993011, 31.7996483
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2751694, 21.2719936
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5942841, 34.5920296
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8574371, 33.8587608
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2140961, 36.2123795
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7569504, 25.7551460
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3199234, 34.3160591
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4799042, 24.4808426
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3284836, 33.3274841
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4568100, 52.4553528
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3434525, 24.3455906
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4758606, 21.4779778
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6874008, 19.6877441
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3389969, 31.3390121
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3663483, 22.3664742
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3953514, 21.3998032
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3553581, 21.3561478
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6548843, 37.6580582
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3940201, 24.3961830
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9647980, 23.9658356
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7815628, 35.7822342
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7418823, 25.7419472
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0674515, 25.0694180
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3465881, 22.3449745
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2224960, 32.2247925
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8260841, 27.8283005
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2446594, 26.2476349
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5352249, 24.5385704
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5753555, 32.5803604
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7338638, 27.7397766
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4284286, 34.4325485
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7499199, 31.7533493
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0196152, 20.0245895
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6392536, 19.6399269

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8448230, upper bound: 17.8640843
time: 19.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8605956, upper bound: 17.8495334
time: 30.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9918251, 19.0038910
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7998047, 31.7991447
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2765846, 21.2705803
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5984726, 34.5878448
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8583832, 33.8578072
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2153015, 36.2111702
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7612686, 25.7508202
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3255310, 34.3104477
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4819260, 24.4788208
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3261108, 33.3298492
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4635696, 52.4486008
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3395767, 24.3493099
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4759674, 21.4778137
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6858444, 19.6892509
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3423157, 31.3356857
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3658981, 22.3669281
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3959770, 21.3991776
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3565788, 21.3549232
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6550980, 37.6578903
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3908920, 24.3995094
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9657211, 23.9649162
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7847672, 35.7790298
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7434998, 25.7403297
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0687943, 25.0680733
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3438110, 22.3477516
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2187347, 32.2285576
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8220253, 27.8323593
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2434540, 26.2488441
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5325546, 24.5412521
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5709610, 32.5847549
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7278671, 27.7457809
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4256210, 34.4353600
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7452583, 31.7580147
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0149803, 20.0292244
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6340389, 19.6448612

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8642743, upper bound: 17.8403622
time: 29.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8841707, upper bound: 17.8270277
time: 21.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9947739, 19.0009441
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7984161, 31.8005257
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2813835, 21.2657814
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5984573, 34.5878525
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8564529, 33.8597412
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2189789, 36.2074966
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7651443, 25.7469444
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3247681, 34.3112144
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4798355, 24.4809113
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3308182, 33.3251457
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4630356, 52.4491348
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3374786, 24.3515663
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4726562, 21.4811783
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6850967, 19.6900558
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3381348, 31.3398743
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3654404, 22.3673782
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3902321, 21.4049225
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3552284, 21.3562737
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6514740, 37.6614609
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3871536, 24.4030495
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9631500, 23.9674911
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7812500, 35.7825546
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7404785, 25.7433510
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0645676, 25.0723038
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3468475, 22.3447189
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2225189, 32.2247772
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8228722, 27.8315125
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2447052, 26.2475929
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5340500, 24.5397491
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5741501, 32.5815697
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7287292, 27.7449131
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4285507, 34.4324265
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7485237, 31.7547493
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0175629, 20.0266418
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6409588, 19.6382256

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8390951
time: 21.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8855296, upper bound: 17.8257550
time: 34.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0009422, 18.9947739
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8005295, 31.7984161
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2657814, 21.2813835
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5878525, 34.5984650
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8597412, 33.8564568
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2074966, 36.2189789
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7469406, 25.7651443
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3112183, 34.3247643
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4809113, 24.4798355
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3251419, 33.3308182
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4491348, 52.4630356
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3515701, 24.3374748
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4811707, 21.4726562
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6900558, 19.6850967
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3398743, 31.3381348
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3673782, 22.3654442
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4049187, 21.3902321
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3562737, 21.3552284
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6614609, 37.6514740
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4030533, 24.3871574
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9674911, 23.9631500
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7825546, 35.7812500
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7433548, 25.7404823
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0723038, 25.0645657
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3447189, 22.3468475
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2247772, 32.2225189
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8315086, 27.8228760
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2475891, 26.2447052
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5397415, 24.5340538
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5815659, 32.5741501
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7449112, 27.7287312
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4324265, 34.4285507
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7547493, 31.7485237
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0266418, 20.0175629
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6382275, 19.6409569

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8257550, upper bound: 17.8855296
time: 24.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8390950, upper bound: 17.8656474
time: 20.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0038910, 18.9918251
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7991486, 31.7998009
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2705803, 21.2765827
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5878372, 34.5984726
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8578110, 33.8583870
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2111664, 36.2153015
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7508163, 25.7612724
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3104477, 34.3255310
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4788208, 24.4819260
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3298492, 33.3261108
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4486008, 52.4635696
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3493118, 24.3395729
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4778137, 21.4759598
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6892471, 19.6858482
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3356857, 31.3423233
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3669281, 22.3658943
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3991814, 21.3959770
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3549232, 21.3565826
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6578903, 37.6550980
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3995132, 24.3908958
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9649200, 23.9657211
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7790298, 35.7847672
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7403336, 25.7434998
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0680771, 25.0687962
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3477554, 22.3438110
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2285538, 32.2187347
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8323555, 27.8220253
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2488403, 26.2434502
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5412521, 24.5325508
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5847549, 32.5709610
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7457809, 27.7278652
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4353561, 34.4256172
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7580147, 31.7452583
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0292244, 20.0149803
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6448612, 19.6340370

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8841707
time: 28.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8270277, upper bound: 17.8642743
time: 18.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9998169, 18.9959011
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7996521, 31.7992935
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2719917, 21.2751694
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5920258, 34.5942841
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8587570, 33.8574371
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2123795, 36.2140961
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7551498, 25.7569466
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3160553, 34.3199196
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4808426, 24.4799080
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3274841, 33.3284836
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4553452, 52.4568176
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3455925, 24.3434505
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4779816, 21.4758568
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6877441, 19.6874046
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3390121, 31.3389969
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3664780, 22.3663483
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3997993, 21.3953552
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3561516, 21.3553543
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6580582, 37.6548843
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3961868, 24.3940239
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9658356, 23.9647980
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7822342, 35.7815628
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7419510, 25.7418823
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0694199, 25.0674515
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3449707, 22.3465881
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2247925, 32.2224998
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8282967, 27.8260880
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2476349, 26.2446632
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5385666, 24.5352287
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5803604, 32.5753555
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7397766, 27.7338676
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4325485, 34.4284286
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7533531, 31.7499237
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0245895, 20.0196152
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6399288, 19.6392536

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8495334, upper bound: 17.8605956
time: 25.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8640843, upper bound: 17.8448230
time: 36.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0027657, 18.9929543
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7982712, 31.8006783
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2767944, 21.2703705
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5920181, 34.5942917
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8568268, 33.8593712
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2160492, 36.2104187
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7590103, 25.7530708
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3152924, 34.3206825
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4787521, 24.4819946
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3321915, 33.3237762
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4548264, 52.4573517
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3433380, 24.3455486
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4746170, 21.4791603
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6869431, 19.6881599
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3348236, 31.3431854
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3660202, 22.3667984
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3940544, 21.4011002
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3547935, 21.3567085
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6544876, 37.6585007
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3926468, 24.3977623
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9632645, 23.9673729
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7787170, 35.7850876
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7389297, 25.7448997
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0651855, 25.0716820
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3480072, 22.3435516
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2285767, 32.2187195
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8291512, 27.8252373
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2488861, 26.2434082
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5400772, 24.5337257
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5835495, 32.5721664
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7406387, 27.7330017
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4354858, 34.4254913
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7566185, 31.7466583
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0271721, 20.0170326
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6465664, 19.6323338

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8508013, upper bound: 17.8592223
time: 28.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8403622, upper bound: 17.8434273
time: 26.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9934273, 19.0022888
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8005524, 31.7983894
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2744331, 21.2727299
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5943375, 34.5919762
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8587189, 33.8574715
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2122955, 36.2141800
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7551651, 25.7569237
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3205566, 34.3154259
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4815826, 24.4791679
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3287582, 33.3272095
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4562912, 52.4558792
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3443222, 24.3447189
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4772186, 21.4766083
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6869583, 19.6881905
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3391418, 31.3388748
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3667374, 22.3660889
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3974419, 21.3977089
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3563194, 21.3551788
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6570740, 37.6558609
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3944168, 24.3957863
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9651642, 23.9654694
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7819519, 35.7818451
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7421341, 25.7416954
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0687790, 25.0680885
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3444519, 22.3471146
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2218094, 32.2254868
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8253670, 27.8290138
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2442474, 26.2480545
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5340500, 24.5397530
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5774460, 32.5782700
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7331467, 27.7404976
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4283142, 34.4326630
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7501259, 31.7531471
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0192070, 20.0249996
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6389408, 19.6402435

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8568537, upper bound: 17.8520339
time: 21.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8726459, upper bound: 17.8374811
time: 21.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9963760, 18.9993420
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7991791, 31.7997704
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2792320, 21.2679291
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5943298, 34.5919838
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8567886, 33.8594055
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2159729, 36.2105026
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7590408, 25.7530479
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3197937, 34.3161888
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4794922, 24.4812546
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3334579, 33.3225021
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4557571, 52.4564056
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3420677, 24.3468170
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4738617, 21.4799156
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6861496, 19.6889458
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3349533, 31.3430557
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3662796, 22.3665390
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3917046, 21.4034538
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3549690, 21.3565331
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6535110, 37.6594772
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3908768, 24.3995247
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9625931, 23.9680405
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7784348, 35.7853699
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7391205, 25.7447128
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0645523, 25.0723190
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3474884, 22.3440781
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2255859, 32.2217064
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8262215, 27.8281631
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2454987, 26.2467995
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5355453, 24.5382500
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5806351, 32.5750809
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7340164, 27.7396317
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4312515, 34.4297256
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7533913, 31.7498817
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0217857, 20.0224152
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6455746, 19.6333237

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8581190, upper bound: 17.8506593
time: 19.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8739070, upper bound: 17.8360916
time: 18.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9922981, 19.0034180
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7996826, 31.7992668
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2806473, 21.2665176
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5985184, 34.5877991
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8577423, 33.8584557
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2171783, 36.2092972
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7633591, 25.7487221
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3254013, 34.3105774
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4815063, 24.4792404
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3310928, 33.3248711
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4625168, 52.4496613
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3383484, 24.3506947
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4740295, 21.4798088
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6846466, 19.6905022
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3382721, 31.3397369
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3658295, 22.3669930
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3923225, 21.4028320
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3561974, 21.3553047
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6536713, 37.6592636
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3875504, 24.4026527
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9635162, 23.9671249
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7816391, 35.7821655
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7407379, 25.7430954
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0658951, 25.0709724
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3447113, 22.3468552
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2218246, 32.2254715
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8221626, 27.8322258
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2442932, 26.2480125
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5328751, 24.5409279
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5762405, 32.5794754
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7280121, 27.7456360
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4284439, 34.4325371
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7487297, 31.7545433
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0171547, 20.0270519
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6406422, 19.6385422

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8777204, upper bound: 17.8270518
time: 25.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8976116, upper bound: 17.8137332
time: 19.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9952469, 19.0004692
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7982941, 31.8006477
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2854462, 21.2617188
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5985031, 34.5878067
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8558044, 33.8603859
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2208557, 36.2056198
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7672348, 25.7448502
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3246307, 34.3113441
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4794235, 24.4813271
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3357925, 33.3201675
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4619675, 52.4501953
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3360901, 24.3527927
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4706573, 21.4831161
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6838455, 19.6912575
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3340912, 31.3439178
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3653793, 22.3674431
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3865776, 21.4085770
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3548470, 21.3566589
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6501007, 37.6628876
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3840103, 24.4063873
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9609451, 23.9696960
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7781143, 35.7856827
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7377167, 25.7461128
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0616684, 25.0752029
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3477478, 22.3438187
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2256012, 32.2216911
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8230095, 27.8313751
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2455444, 26.2467575
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5343704, 24.5394249
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5794296, 32.5762863
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7288742, 27.7447681
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4313736, 34.4296036
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7519951, 31.7512817
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0197334, 20.0244675
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6472797, 19.6316223

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8789810, upper bound: 17.8256596
time: 19.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8988680, upper bound: 17.8123254
time: 21.27 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 42.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8123254, upper bound: 17.8988680
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8256596, upper bound: 17.8789810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8976116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8270518, upper bound: 17.8777205
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8360916, upper bound: 17.8739071
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8506593, upper bound: 17.8581190
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8726459
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8520339, upper bound: 17.8568537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8434273, upper bound: 17.8653452
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8592223, upper bound: 17.8508013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8448230, upper bound: 17.8640843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8605956, upper bound: 17.8495334
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8642743, upper bound: 17.8403622
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8841707, upper bound: 17.8270277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8390951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8855296, upper bound: 17.8257550
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8257550, upper bound: 17.8855296
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8390950, upper bound: 17.8656474
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8841707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8270277, upper bound: 17.8642743
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8495334, upper bound: 17.8605956
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8640843, upper bound: 17.8448230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8508013, upper bound: 17.8592223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8403622, upper bound: 17.8434273
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8568537, upper bound: 17.8520339
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8726459, upper bound: 17.8374811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8581190, upper bound: 17.8506593
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8739070, upper bound: 17.8360916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8777204, upper bound: 17.8270518
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8976116, upper bound: 17.8137332
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8789810, upper bound: 17.8256596
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 42.64
Output dim: 10, lower bound: -17.8988680, upper bound: 17.8123254

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9876442, 18.9807415
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8063202, 31.8029480
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2267647, 21.2561913
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5655212, 34.5798569
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8524590, 33.8463326
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2162476, 36.2358017
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7030716, 25.7322655
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2785950, 34.2975006
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4831772, 24.4810410
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3067551, 33.3244743
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4438171, 52.4605179
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3255959, 24.3035946
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4675827, 21.4521141
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6835403, 19.6746140
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3412704, 31.3310394
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3583679, 22.3545380
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3794098, 21.3517342
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3544998, 21.3524017
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6451340, 37.6287994
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3739700, 24.3452873
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9603806, 23.9498138
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7825165, 35.7744446
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7390671, 25.7292900
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0616531, 25.0454750
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3477936, 22.3523712
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2193298, 32.2227325
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8373642, 27.8253670
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2453690, 26.2434196
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5284805, 24.5215187
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5527954, 32.5529404
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7344131, 27.7127876
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4306870, 34.4315643
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7428932, 31.7409744
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0022507, 19.9944706
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6353951, 19.6518173

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8099304, upper bound: 17.8812193
time: 21.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7946628, upper bound: 17.8965503
time: 23.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9859657, 18.9824600
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8052979, 31.8039665
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2324638, 21.2504940
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5691528, 34.5762253
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8509178, 33.8478775
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2205658, 36.2314758
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7098846, 25.7254562
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2835693, 34.2918816
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4829483, 24.4812508
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3088455, 33.3223801
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4487305, 52.4556046
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3202972, 24.3088913
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4645615, 21.4551353
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6820297, 19.6761169
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3408661, 31.3314362
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3565979, 22.3563042
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3737335, 21.3574142
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3542099, 21.3526840
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6415863, 37.6323547
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3676682, 24.3516006
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9585648, 23.9516296
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7820129, 35.7749481
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7376938, 25.7306633
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0590134, 25.0481129
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3484497, 22.3517151
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2188187, 32.2232895
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8337326, 27.8289986
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2446365, 26.2441444
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5265732, 24.5234871
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5497971, 32.5554581
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7286758, 27.7185173
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4297943, 34.4323692
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7402611, 31.7431984
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9992027, 19.9975166
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6361732, 19.6510506

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8232972, upper bound: 17.8613236
time: 20.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8079756, upper bound: 17.8766296
time: 16.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9905891, 18.9777946
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8049316, 31.8043289
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2315636, 21.2513924
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5655136, 34.5798645
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8505211, 33.8482666
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2199173, 36.2321243
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7069397, 25.7283936
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2778320, 34.2982674
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4810867, 24.4831276
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3114624, 33.3197708
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4432983, 52.4610519
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3234940, 24.3058510
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4642868, 21.4554749
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6827774, 19.6754189
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3370819, 31.3352280
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3579178, 22.3549843
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3736725, 21.3574791
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3531418, 21.3537560
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6415100, 37.6323700
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3702316, 24.3488274
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9578171, 23.9523849
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7789917, 35.7779694
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7360458, 25.7323074
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0574188, 25.0497055
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3508301, 22.3493347
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2231064, 32.2189522
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8382111, 27.8245163
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2466354, 26.2421684
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5299835, 24.5200157
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5559845, 32.5497513
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7352753, 27.7119217
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4336243, 34.4286270
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7461586, 31.7377090
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0048332, 19.9918880
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6423149, 19.6451797

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8113450, upper bound: 17.8799615
time: 19.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7960758, upper bound: 17.8952888
time: 16.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9889107, 18.9795113
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8039169, 31.8053474
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2372627, 21.2456932
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5691452, 34.5762329
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8489799, 33.8498116
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2242432, 36.2277985
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7137527, 25.7215805
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2827988, 34.2926483
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4808655, 24.4833412
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3135529, 33.3176765
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4482117, 52.4561386
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3181992, 24.3111477
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4612579, 21.4584961
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6812744, 19.6769218
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3366852, 31.3356247
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3561554, 22.3567505
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3679962, 21.3631592
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3528671, 21.3540344
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6379700, 37.6359177
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3639297, 24.3551407
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9560013, 23.9542007
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7784882, 35.7784729
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7346725, 25.7336845
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0547791, 25.0523434
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3514862, 22.3486786
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2225952, 32.2195129
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8345795, 27.8281479
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2458878, 26.2428932
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5280762, 24.5219841
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5529861, 32.5522690
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7295456, 27.7176495
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4327240, 34.4294357
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7435265, 31.7399330
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0017853, 19.9949341
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6430931, 19.6444130

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8093736, upper bound: 17.8600584
time: 18.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8093736, upper bound: 17.8753618
time: 20.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9865532, 18.9818687
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8054352, 31.8038254
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2329788, 21.2499790
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5696945, 34.5756798
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8514824, 33.8473167
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2211304, 36.2309189
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7112732, 25.7240677
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2834396, 34.2920074
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4831085, 24.4811134
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3090897, 33.3221397
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4500427, 52.4542999
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3196182, 24.3095703
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4643860, 21.4553146
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6812210, 19.6769257
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3404007, 31.3319016
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3574677, 22.3554382
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3742828, 21.3568573
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3543701, 21.3525276
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6417313, 37.6322098
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3671036, 24.3521538
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9587326, 23.9514656
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7821960, 35.7747650
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7376633, 25.7306900
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0587692, 25.0483589
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3480530, 22.3521156
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2194138, 32.2227135
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8341522, 27.8285789
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2454147, 26.2433739
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5273590, 24.5226974
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5516052, 32.5541458
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7292709, 27.7179241
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4308167, 34.4314384
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7414970, 31.7423744
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0001984, 19.9965229
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6370964, 19.6501503

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8337166, upper bound: 17.8562524
time: 31.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8183969, upper bound: 17.8716066
time: 25.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9848366, 18.9835491
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8044205, 31.8048439
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2386742, 21.2442818
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5733337, 34.5720444
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8499413, 33.8488617
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2254486, 36.2265930
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7180786, 25.7172546
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2890625, 34.2870331
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4828796, 24.4813194
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3111801, 33.3200455
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4549561, 52.4493866
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3143196, 24.3148670
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4613647, 21.4583359
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6797180, 19.6784286
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3400040, 31.3322983
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3556976, 22.3572083
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3686066, 21.3625374
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3540878, 21.3528099
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6381836, 37.6357574
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3608017, 24.3584671
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9569168, 23.9532814
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7816925, 35.7752686
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7362900, 25.7320671
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0561295, 25.0509987
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3487091, 22.3514557
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2188339, 32.2232056
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8305206, 27.8322067
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2446823, 26.2441139
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5253983, 24.5246048
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5485916, 32.5566483
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7235413, 27.7236538
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4299164, 34.4322433
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7388649, 31.7445984
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9971504, 19.9995689
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6378403, 19.6493473

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1356

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8483031, upper bound: 17.8404573
time: 20.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8329672, upper bound: 17.8558324
time: 35.26 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 58.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8099304, upper bound: 17.8812193
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.7946628, upper bound: 17.8965503
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8232972, upper bound: 17.8613236
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8079756, upper bound: 17.8766296
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8113450, upper bound: 17.8799615
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.7960758, upper bound: 17.8952888
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8093736, upper bound: 17.8600584
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8093736, upper bound: 17.8753618
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8337166, upper bound: 17.8562524
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8183969, upper bound: 17.8716066
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8483031, upper bound: 17.8404573
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 58.22
Output dim: 10, lower bound: -17.8329672, upper bound: 17.8558324
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8726459
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8520339, upper bound: 17.8568537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8434273, upper bound: 17.8653452
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8592223, upper bound: 17.8508013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8448230, upper bound: 17.8640843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8605956, upper bound: 17.8495334
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8642743, upper bound: 17.8403622
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8841707, upper bound: 17.8270277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8390951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8855296, upper bound: 17.8257550
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8257550, upper bound: 17.8855296
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8390950, upper bound: 17.8656474
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8841707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8270277, upper bound: 17.8642743
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8495334, upper bound: 17.8605956
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8640843, upper bound: 17.8448230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8508013, upper bound: 17.8592223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8403622, upper bound: 17.8434273
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8568537, upper bound: 17.8520339
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8726459, upper bound: 17.8374811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8581190, upper bound: 17.8506593
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8739070, upper bound: 17.8360916
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8777204, upper bound: 17.8270518
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8976116, upper bound: 17.8137332
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8789810, upper bound: 17.8256596
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 58.22
Output dim: 10, lower bound: -17.8988680, upper bound: 17.8123254

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 32.13 + 1810.46 = 1842.59 seconds
