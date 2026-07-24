## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 17.8846163811


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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
execution time: IAR + RelationalAnalysis = 2.71 + 29.55 = 32.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.9025189, upper bound: 17.9025189

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1361

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 515

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9023272, upper bound: 17.8983794
time: 30.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8983794, upper bound: 17.9023272
time: 22.18 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 52.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 52.58
Output dim: 10, lower bound: -17.9023272, upper bound: 17.8983794
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 52.58
Output dim: 10, lower bound: -17.8983794, upper bound: 17.9023272

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0118408, 19.0131493
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7947845, 31.7937813
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4041748, 21.4041061
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6690903, 34.6680984
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8889694, 33.8886642
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1791534, 36.1794739
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8892021, 25.8899612
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3956490, 34.3936081
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4712639, 24.4700317
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3631897, 33.3632507
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4836197, 52.4822311
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4216003, 24.4217587
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5478058, 21.5479584
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7430191, 19.7429314
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3955307, 31.3947678
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3792114, 22.3793068
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5215302, 21.5211487
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3778534, 21.3760262
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7260971, 37.7255478
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5363045, 24.5362663
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0137177, 24.0137558
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8257599, 35.8248138
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7952957, 25.7937202
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1399307, 25.1396446
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3125916, 22.3146095
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2714691, 32.2736015
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8420219, 27.8438568
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2649002, 26.2668228
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5777550, 24.5799599
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6488495, 32.6505775
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7960472, 27.7991505
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4779968, 34.4804955
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8064117, 31.8077393
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0594215, 20.0620689
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5597267, 19.5599747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1438

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1756

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8910259, upper bound: 17.8975079
time: 21.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9014560, upper bound: 17.8870773
time: 17.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0119476, 19.0118408
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7937775, 31.7938194
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4041061, 21.4043617
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6680908, 34.6689186
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8886642, 33.8888054
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1794128, 36.1791534
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8892784, 25.8892021
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3936043, 34.3938103
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4700279, 24.4708939
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3632507, 33.3633156
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4822311, 52.4835129
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4217415, 24.4216003
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5478134, 21.5478096
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7429276, 19.7429657
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3947678, 31.3948288
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3794556, 22.3792114
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5211487, 21.5211945
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3760300, 21.3764000
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7255478, 37.7256012
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5362587, 24.5362701
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0137558, 24.0137749
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8248062, 35.8249359
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7937241, 25.7941742
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1396408, 25.1398239
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3143997, 22.3125877
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2729187, 32.2714691
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8428612, 27.8420181
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2664719, 26.2649040
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5797386, 24.5777473
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6503754, 32.6488457
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7989540, 27.7960472
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4803543, 34.4779968
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8075867, 31.8064079
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0617027, 20.0594215
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5602455, 19.5597267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 516

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8960635, upper bound: 17.8846643
time: 19.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8807159, upper bound: 17.9000130
time: 17.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 38.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 38.41
Output dim: 10, lower bound: -17.8910259, upper bound: 17.8975079
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 38.41
Output dim: 10, lower bound: -17.9014560, upper bound: 17.8870773
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 38.41
Output dim: 10, lower bound: -17.8960635, upper bound: 17.8846643
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 38.41
Output dim: 10, lower bound: -17.8807159, upper bound: 17.9000130

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0096550, 19.0122452
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7949066, 31.7914391
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3870850, 21.3928871
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6679382, 34.6671448
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8871841, 33.8839226
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1695175, 36.1744156
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8662071, 25.8747711
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3943977, 34.3923683
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4723816, 24.4687805
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3507004, 33.3548508
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4773636, 52.4773331
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4159012, 24.4118938
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5431671, 21.5407944
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7409363, 19.7402382
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3931122, 31.3916473
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3712883, 22.3671417
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5110703, 21.5050812
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3752708, 21.3726692
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7200699, 37.7154160
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5243073, 24.5182838
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0098267, 24.0081215
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8222046, 35.8192749
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7934418, 25.7917900
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1339264, 25.1306763
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3121414, 22.3156471
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2686005, 32.2717972
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8367081, 27.8359795
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2646179, 26.2664948
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5777397, 24.5799484
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6468048, 32.6485977
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7899857, 27.7902622
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4751587, 34.4782181
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8050117, 31.8064117
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0564842, 20.0580921
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5593109, 19.5597839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1395

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1641

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8764965, upper bound: 17.8963207
time: 19.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8898390, upper bound: 17.8829768
time: 18.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0109367, 19.0109653
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7924423, 31.7939034
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3929596, 21.3870049
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6681366, 34.6669502
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8842239, 33.8868790
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1740952, 36.1698380
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8740120, 25.8669662
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3943977, 34.3923645
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4700241, 24.4711456
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3547897, 33.3507652
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4787369, 52.4759750
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4117355, 24.4160576
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5406494, 21.5433159
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7403259, 19.7408524
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3924179, 31.3923492
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3670464, 22.3713913
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5054626, 21.5106888
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3744926, 21.3734512
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7159653, 37.7195282
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5183258, 24.5242653
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0080872, 24.0098610
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8202286, 35.8212509
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7933655, 25.7918663
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1309662, 25.1336327
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3136215, 22.3141670
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2696686, 32.2707329
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8341446, 27.8385468
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2645798, 26.2665367
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5777397, 24.5799561
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6468658, 32.6485367
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7871628, 27.7930870
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4757233, 34.4776611
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8050804, 31.8063469
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0554466, 20.0591373
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5595360, 19.5595589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1320

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1363

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9007463, upper bound: 17.8863494
time: 25.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9007288, upper bound: 17.8863668
time: 26.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0124702, 19.0098724
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7930298, 31.7910957
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4034119, 21.4040642
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6673737, 34.6679230
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8886642, 33.8888054
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1788177, 36.1786842
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8801537, 25.8845100
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933792, 34.3934250
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4700775, 24.4708099
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3624420, 33.3624954
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4821548, 52.4834137
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4199448, 24.4189968
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5477753, 21.5477905
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7432556, 19.7424431
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3942795, 31.3944016
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3793564, 22.3790855
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5208817, 21.5209732
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3748779, 21.3751602
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7260818, 37.7255020
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5351295, 24.5332909
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0135803, 24.0136719
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243713, 35.8240662
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7937622, 25.7934494
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1395645, 25.1397514
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3143845, 22.3125725
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2690659, 32.2694206
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8419685, 27.8413734
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2618332, 26.2625275
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5752068, 24.5754242
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6497955, 32.6485062
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7954140, 27.7939339
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4722672, 34.4737244
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8087311, 31.8061867
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0611286, 20.0588341
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5593796, 19.5576878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1363

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1406

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8945711, upper bound: 17.8816710
time: 22.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8930692, upper bound: 17.8831753
time: 36.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0099831, 19.0123596
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7910690, 31.7930565
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4038086, 21.4036713
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6670914, 34.6682053
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8886642, 33.8888092
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1789551, 36.1785469
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8845940, 25.8800697
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3932114, 34.3935852
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4699402, 24.4709473
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3624268, 33.3625107
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4821243, 52.4834442
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4191360, 24.4198055
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5477905, 21.5477676
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7424011, 19.7432976
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3943405, 31.3943405
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3793259, 22.3791199
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5209274, 21.5209312
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3747864, 21.3752480
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7254486, 37.7261353
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5332832, 24.5351410
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0136490, 24.0136032
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8239288, 35.8245010
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7929993, 25.7942238
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1395645, 25.1397533
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3143845, 22.3125725
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2708740, 32.2676163
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8422279, 27.8411102
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2640762, 26.2602806
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5774040, 24.5732231
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6500397, 32.6482620
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7968483, 27.7925091
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4760895, 34.4698982
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8073578, 31.8075562
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0611172, 20.0588455
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5582085, 19.5588608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1395

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8804602, upper bound: 17.8929108
time: 21.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8736123, upper bound: 17.8997573
time: 22.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 46.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.8764965, upper bound: 17.8963207
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.8898390, upper bound: 17.8829768
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.9007463, upper bound: 17.8863494
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.9007288, upper bound: 17.8863668
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.8945711, upper bound: 17.8816710
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.8930692, upper bound: 17.8831753
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.8804602, upper bound: 17.8929108
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 46.62
Output dim: 10, lower bound: -17.8736123, upper bound: 17.8997573

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0202179, 19.0232773
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7942963, 31.7907104
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3594017, 21.3692703
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6558914, 34.6551437
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8821068, 33.8782043
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1717758, 36.1785507
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8434525, 25.8541107
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3915634, 34.3894043
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4763336, 24.4723282
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3535080, 33.3626328
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4682770, 52.4671860
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4126358, 24.4072495
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5309639, 21.5265961
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7284851, 19.7265396
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3728256, 31.3673096
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3754997, 22.3712807
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4910088, 21.4813652
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3690643, 21.3660812
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7108154, 37.7047882
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5027351, 24.4935570
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9987717, 23.9948616
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8065262, 35.8004684
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7795868, 25.7751694
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1174202, 25.1112709
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3208351, 22.3252411
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2497482, 32.2560272
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8342514, 27.8336563
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2608032, 26.2635078
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5763359, 24.5788727
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6122055, 32.6192780
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7887535, 27.7891788
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4581985, 34.4640846
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7800598, 31.7849236
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0571423, 20.0609188
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5834827, 19.5905571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1330

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8762296, upper bound: 17.8961449
time: 28.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8763212, upper bound: 17.8960522
time: 30.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0206909, 19.0228043
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7941742, 31.7908325
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3634605, 21.3652058
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6559372, 34.6550980
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8814583, 33.8788528
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1736526, 36.1766739
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8455505, 25.8520203
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3914337, 34.3895340
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4759216, 24.4727440
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3584900, 33.3576546
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4672089, 52.4682465
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4112549, 24.4086342
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5289650, 21.5285912
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7272339, 19.7277908
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3687820, 31.3713608
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3754387, 22.3713455
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4873543, 21.4850197
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3686829, 21.3664627
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7094421, 37.7061539
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4995918, 24.4967003
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9965668, 23.9970703
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8033905, 35.8035965
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7768097, 25.7779350
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1145210, 25.1141720
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3217354, 22.3243446
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2528305, 32.2529411
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8343887, 27.8335228
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2616348, 26.2626724
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5766563, 24.5785484
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6174850, 32.6139984
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7888985, 27.7890339
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4610214, 34.4612617
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7835312, 31.7814560
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0593128, 20.0587463
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5900860, 19.5839539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1410

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1391

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8722544, upper bound: 17.8826806
time: 27.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8895452, upper bound: 17.8826806
time: 17.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0091095, 19.0088482
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7924347, 31.7938957
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3927803, 21.3867531
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6679459, 34.6667633
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8840179, 33.8867569
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1737289, 36.1693649
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8735886, 25.8663673
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3935928, 34.3914261
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4690666, 24.4704056
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3546143, 33.3505135
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4783630, 52.4753876
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4115982, 24.4159603
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5404816, 21.5432014
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7401199, 19.7407074
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3918991, 31.3920441
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3670387, 22.3713875
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5049019, 21.5103645
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3736496, 21.3729782
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7156677, 37.7193527
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5181503, 24.5241432
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0080414, 24.0097313
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8198700, 35.8210373
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7929382, 25.7916145
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306648, 25.1334610
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3127213, 22.3128700
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2691879, 32.2701111
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8333740, 27.8376007
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2644424, 26.2663536
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5774422, 24.5795288
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6463699, 32.6481628
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7863312, 27.7919769
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4756470, 34.4775543
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8043175, 31.8057899
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0539665, 20.0572643
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5583000, 19.5582542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1316

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 514

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9003750, upper bound: 17.8850928
time: 18.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8988408, upper bound: 17.8861090
time: 26.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0088196, 19.0091381
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7924347, 31.7938957
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3927040, 21.3868256
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6679535, 34.6667595
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8840942, 33.8866692
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1736221, 36.1694717
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8734131, 25.8665428
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3934708, 34.3915443
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4692879, 24.4701920
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3545380, 33.3505859
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4781494, 52.4756088
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4116364, 24.4159222
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5405273, 21.5431557
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7401733, 19.7406502
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3921051, 31.3918304
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3670387, 22.3713837
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5051384, 21.5101242
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3740158, 21.3726044
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7157898, 37.7192307
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5182037, 24.5240860
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0079575, 24.0098190
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8200150, 35.8208923
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7931213, 25.7914314
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1307945, 25.1333351
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3123245, 22.3132591
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2690506, 32.2702522
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8331985, 27.8377762
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2643967, 26.2663994
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5773125, 24.5796623
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6464920, 32.6480370
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7860413, 27.7922573
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4756165, 34.4775772
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8045235, 31.8055840
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0535698, 20.0576611
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5582314, 19.5583267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1322

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9000233, upper bound: 17.8857266
time: 19.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9000878, upper bound: 17.8856615
time: 24.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0124702, 19.0098724
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7929382, 31.7908897
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4031906, 21.4034958
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6673203, 34.6677666
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8884354, 33.8887138
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1788635, 36.1785889
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8797531, 25.8842125
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933945, 34.3933411
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4700317, 24.4706650
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3623199, 33.3621712
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4821548, 52.4834137
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4194641, 24.4188251
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5474625, 21.5476685
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7430420, 19.7423363
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3940811, 31.3943253
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3792725, 22.3790512
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5203476, 21.5207596
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3746033, 21.3750458
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7257156, 37.7253647
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5349464, 24.5331078
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0133591, 24.0135841
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8241119, 35.8239594
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7934875, 25.7933540
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1392899, 25.1396427
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3144417, 22.3123970
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2691803, 32.2691879
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8419228, 27.8413582
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2617722, 26.2624969
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5751152, 24.5753784
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6499100, 32.6482964
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7949715, 27.7937565
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4722824, 34.4735298
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8088837, 31.8058853
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0610657, 20.0587044
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5595303, 19.5574799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1439

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8944168, upper bound: 17.8803460
time: 26.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8932395, upper bound: 17.8815222
time: 26.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0124702, 19.0098724
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7928085, 31.7910194
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4028549, 21.4038353
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6672287, 34.6678505
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8885727, 33.8885727
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1787186, 36.1787338
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8798523, 25.8841095
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3932877, 34.3934441
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4699402, 24.4707565
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3621292, 33.3623619
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4821548, 52.4834137
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4197769, 24.4185143
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5476532, 21.5474777
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7431488, 19.7422295
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3942032, 31.3942032
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3793259, 22.3789978
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5206757, 21.5204391
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3747635, 21.3748856
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7259369, 37.7251434
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5349541, 24.5330963
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0134964, 24.0134468
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8242645, 35.8238068
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7936707, 25.7931709
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1394577, 25.1394768
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3142128, 22.3126221
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2688293, 32.2695389
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8419533, 27.8413391
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2618179, 26.2624550
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5751610, 24.5753212
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6495895, 32.6486168
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7952385, 27.7934875
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4720688, 34.4737473
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8084335, 31.8063393
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0609970, 20.0587749
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5591717, 19.5578384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1439

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1314

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8842975, upper bound: 17.8744774
time: 22.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8842975, upper bound: 17.8744774
time: 22.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0043106, 19.0064087
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7925797, 31.7942390
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4041443, 21.4039211
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6665649, 34.6676025
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8894386, 33.8897400
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1774521, 36.1768723
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8818893, 25.8770828
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933296, 34.3936653
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4695663, 24.4705696
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3631363, 33.3630447
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4813690, 52.4825592
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4193382, 24.4200687
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5484352, 21.5486031
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7416191, 19.7426338
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3947830, 31.3949432
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3802643, 22.3800087
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5221176, 21.5224953
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3764954, 21.3771210
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7262344, 37.7271194
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5311699, 24.5333405
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0145111, 24.0144997
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243866, 35.8249054
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7930336, 25.7942734
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1400719, 25.1404133
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3108749, 22.3089523
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2683945, 32.2650757
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8410225, 27.8399887
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2649384, 26.2614098
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5779610, 24.5739517
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6473274, 32.6456528
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7973099, 27.7931442
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4761581, 34.4699860
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8012848, 31.8006554
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0581207, 20.0557308
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5541878, 19.5542374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1426

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1347

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8793401, upper bound: 17.8926076
time: 26.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8801600, upper bound: 17.8917892
time: 30.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0040283, 19.0066872
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7922287, 31.7945938
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4040680, 21.4039993
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6664810, 34.6676941
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8895988, 33.8895798
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1772766, 36.1770515
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8815994, 25.8773651
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3932991, 34.3936958
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4695663, 24.4705772
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3629684, 33.3632126
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4812469, 52.4826965
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4193993, 24.4200115
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5486336, 21.5484085
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7417412, 19.7425156
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3949509, 31.3947754
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3802032, 22.3800621
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5224915, 21.5221176
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3766632, 21.3769493
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7264404, 37.7269135
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5314827, 24.5330238
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0145569, 24.0144539
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243484, 35.8249435
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7930412, 25.7942619
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1402245, 25.1402588
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3107681, 22.3090630
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2683334, 32.2651443
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8410988, 27.8399086
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2652206, 26.2611351
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5781364, 24.5737801
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6474342, 32.6455460
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7974777, 27.7929859
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4761734, 34.4699707
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8004608, 31.8014793
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0579987, 20.0558548
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5535851, 19.5548401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1306

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8715457, upper bound: 17.8976795
time: 15.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8715480, upper bound: 17.8976773
time: 17.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 36.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8762296, upper bound: 17.8961449
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8763212, upper bound: 17.8960522
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8722544, upper bound: 17.8826806
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8895452, upper bound: 17.8826806
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.9003750, upper bound: 17.8850928
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8988408, upper bound: 17.8861090
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.9000233, upper bound: 17.8857266
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.9000878, upper bound: 17.8856615
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8944168, upper bound: 17.8803460
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8932395, upper bound: 17.8815222
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8842975, upper bound: 17.8744774
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8842975, upper bound: 17.8744774
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8793401, upper bound: 17.8926076
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8801600, upper bound: 17.8917892
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8715457, upper bound: 17.8976795
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.05
Output dim: 10, lower bound: -17.8715480, upper bound: 17.8976773

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0202293, 19.0232811
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7944565, 31.7908134
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3585052, 21.3684826
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6552124, 34.6545181
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8817329, 33.8777771
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1719131, 36.1787796
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8423004, 25.8530807
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3910637, 34.3888435
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4764290, 24.4723740
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3532104, 33.3623657
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4678268, 52.4668045
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4111404, 24.4055538
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5295677, 21.5250931
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7280731, 19.7260399
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3720322, 31.3666077
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3751221, 22.3708496
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4894867, 21.4795990
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3683548, 21.3652382
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7092972, 37.7030563
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5012741, 24.4919052
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9981232, 23.9941216
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8059387, 35.7997971
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7790909, 25.7747192
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1162529, 25.1099396
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3210564, 22.3254623
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2497177, 32.2560005
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8342781, 27.8336678
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2601395, 26.2627296
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5758209, 24.5783157
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6117325, 32.6187592
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7871819, 27.7873840
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4574051, 34.4632416
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7801285, 31.7849388
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0571556, 20.0609188
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5837097, 19.5906105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1300

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8677400, upper bound: 17.8875337
time: 28.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8677193, upper bound: 17.8875444
time: 18.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0202217, 19.0232925
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7943954, 31.7908745
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3586121, 21.3683739
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6552734, 34.6544647
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8816795, 33.8778305
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1720047, 36.1786880
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8424225, 25.8529587
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3910103, 34.3889008
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4763908, 24.4724121
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3532410, 33.3623352
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4679031, 52.4667358
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4109421, 24.4057503
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5294609, 21.5252037
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7279816, 19.7261238
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3721237, 31.3665237
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3750610, 22.3709030
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4892349, 21.4798470
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3682175, 21.3653679
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7090759, 37.7032700
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5010757, 24.4920998
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9980316, 23.9942093
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8058548, 35.7998810
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7791367, 25.7746849
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1160851, 25.1101112
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3210640, 22.3254547
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2497177, 32.2560005
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8342628, 27.8336792
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2600250, 26.2628441
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5757904, 24.5783577
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6116867, 32.6188011
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7869530, 27.7876129
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4573593, 34.4632874
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7800751, 31.7849960
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0571404, 20.0609303
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5835342, 19.5907860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1455

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1732

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8745342, upper bound: 17.8956285
time: 25.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8758978, upper bound: 17.8942667
time: 40.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0206375, 19.0227814
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7940140, 31.7907333
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3634109, 21.3651848
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6559067, 34.6550407
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8814201, 33.8787880
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1736374, 36.1766701
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8455276, 25.8520050
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3912811, 34.3895073
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4758072, 24.4726257
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3584442, 33.3576355
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4669952, 52.4679184
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4112320, 24.4085808
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5289497, 21.5285568
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7271805, 19.7277069
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3687592, 31.3713226
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3754272, 22.3713303
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4872971, 21.4848824
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3686485, 21.3663826
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7094345, 37.7061462
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4995499, 24.4966583
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9965591, 23.9970512
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8033829, 35.8035507
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7767639, 25.7778587
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1145134, 25.1141548
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3216133, 22.3242683
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2527847, 32.2529526
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8343811, 27.8335152
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2614670, 26.2625427
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5764885, 24.5784225
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6174011, 32.6140137
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7886620, 27.7888813
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4609833, 34.4612656
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7833939, 31.7814903
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0592937, 20.0587349
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5899982, 19.5839462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8548817, upper bound: 17.8817380
time: 27.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8886062, upper bound: 17.8480336
time: 31.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0090904, 19.0091190
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7929001, 31.7938728
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3927803, 21.3866692
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6680450, 34.6665115
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8840637, 33.8867188
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1735687, 36.1693878
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8735657, 25.8667603
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3940582, 34.3913345
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4694138, 24.4701233
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3546143, 33.3505363
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4783478, 52.4749374
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4114723, 24.4159775
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5404739, 21.5433884
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7401085, 19.7406502
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3921890, 31.3919678
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3667641, 22.3712082
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5050659, 21.5103569
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3744736, 21.3727264
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7157745, 37.7193527
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5181503, 24.5241280
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0080185, 24.0097847
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8202057, 35.8209534
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7934303, 25.7911568
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306992, 25.1334000
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3119431, 22.3129997
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2685547, 32.2704163
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8327789, 27.8377953
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2636185, 26.2665787
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5766106, 24.5797119
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6456528, 32.6480904
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7849579, 27.7920380
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4744720, 34.4775734
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8039360, 31.8058243
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0528831, 20.0574436
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5582733, 19.5582104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 595

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1334

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8984455, upper bound: 17.8850003
time: 23.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9002825, upper bound: 17.8831629
time: 22.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0091095, 19.0088272
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7924194, 31.7938957
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3927040, 21.3867531
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6677017, 34.6667633
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8839722, 33.8867569
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1737289, 36.1691971
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8735886, 25.8663483
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3934937, 34.3914261
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4687881, 24.4704056
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3546143, 33.3505173
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4779205, 52.4753876
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4115982, 24.4158325
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5404816, 21.5431900
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7400551, 19.7407074
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3918228, 31.3920441
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3670387, 22.3711128
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5048904, 21.5103645
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3733978, 21.3729782
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7156601, 37.7193527
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5181351, 24.5241432
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0080414, 24.0097046
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8197861, 35.8210373
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7924690, 25.7916145
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306076, 25.1334610
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3127213, 22.3120918
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2691879, 32.2694778
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8333740, 27.8370094
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2644424, 26.2655334
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5774422, 24.5786934
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6463699, 32.6474457
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7863312, 27.7906055
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4756470, 34.4763794
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8043175, 31.8054123
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0539665, 20.0561810
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5583000, 19.5582237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1322

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8981351, upper bound: 17.8854691
time: 24.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8981995, upper bound: 17.8854035
time: 25.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0085602, 19.0089836
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7924118, 31.7936554
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3926506, 21.3866768
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6679916, 34.6666451
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8839417, 33.8864479
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1734619, 36.1694412
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8725586, 25.8661194
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933182, 34.3912697
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4691353, 24.4699745
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3545227, 33.3505402
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4781876, 52.4755478
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4115372, 24.4158134
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5404434, 21.5430565
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7401123, 19.7406540
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3919678, 31.3917770
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3668976, 22.3710976
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5050354, 21.5099716
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3739700, 21.3725586
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7157364, 37.7191315
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5179977, 24.5237694
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0079498, 24.0098076
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8199005, 35.8207550
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7929459, 25.7911987
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306496, 25.1332340
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3122177, 22.3132095
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2685928, 32.2699585
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8331909, 27.8377762
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2639847, 26.2663803
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5770988, 24.5797806
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6463470, 32.6477966
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7857666, 27.7923546
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4747620, 34.4772835
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8042450, 31.8050079
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0533772, 20.0574512
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5578766, 19.5577736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8978835, upper bound: 17.8834855
time: 23.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8978212, upper bound: 17.8835724
time: 30.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0086632, 19.0088806
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7921982, 31.7938728
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3925591, 21.3867683
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6678391, 34.6668015
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8838806, 33.8865166
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1735916, 36.1693192
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8729858, 25.8656845
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3931961, 34.3913994
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4690590, 24.4700432
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3544922, 33.3505707
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4780807, 52.4756546
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4115295, 24.4158211
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5404358, 21.5430641
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7401810, 19.7405815
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3920517, 31.3916931
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3667603, 22.3712387
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5049820, 21.5100212
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3739777, 21.3725548
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7156830, 37.7191772
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5178833, 24.5238876
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0079422, 24.0098114
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8198853, 35.8207703
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7928848, 25.7912598
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306953, 25.1331882
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3122787, 22.3131485
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2687531, 32.2697945
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8331909, 27.8377724
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2643661, 26.2659836
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5774345, 24.5794449
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6462479, 32.6478920
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7861481, 27.7919788
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4753189, 34.4767265
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8039474, 31.8053055
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0533619, 20.0574627
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5576782, 19.5579720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8549751, upper bound: 17.8847328
time: 22.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8549751, upper bound: 17.8509735
time: 29.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0120525, 19.0091858
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7928162, 31.7906914
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4029236, 21.4032745
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6672592, 34.6676865
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8881912, 33.8884544
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1788330, 36.1785278
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8795013, 25.8840637
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3934631, 34.3928375
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4699097, 24.4704781
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3620682, 33.3616066
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4816132, 52.4828796
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4193268, 24.4186440
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5474586, 21.5476570
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7430153, 19.7422142
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3938293, 31.3942032
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3792534, 22.3790131
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5202332, 21.5205994
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3744125, 21.3748016
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7255936, 37.7251282
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5347366, 24.5327072
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0132751, 24.0135536
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8240509, 35.8237610
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7933960, 25.7931633
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1392136, 25.1394844
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3140831, 22.3120651
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2685165, 32.2685890
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8418732, 27.8413277
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2615738, 26.2623825
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5750771, 24.5753593
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6486206, 32.6464539
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7938690, 27.7928009
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4716034, 34.4730377
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8079834, 31.8046036
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0608368, 20.0582428
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5594559, 19.5573483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1651

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8943498, upper bound: 17.8791982
time: 18.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8932696, upper bound: 17.8802841
time: 26.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0117855, 19.0094566
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7927475, 31.7907600
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4029694, 21.4032307
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6672211, 34.6677170
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8881760, 33.8884773
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1788025, 36.1785583
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8796082, 25.8839645
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3928986, 34.3934059
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4698486, 24.4705391
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3617630, 33.3619156
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4816284, 52.4828720
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4192848, 24.4186859
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5474510, 21.5476608
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7429237, 19.7422981
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3939514, 31.3940811
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3792305, 22.3790359
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5201950, 21.5206451
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3743515, 21.3748589
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7254944, 37.7252274
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5345459, 24.5328979
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0133286, 24.0135002
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8239288, 35.8238831
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7932892, 25.7932739
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1391373, 25.1395626
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3141136, 22.3120384
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2685699, 32.2685394
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8418961, 27.8413048
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2616501, 26.2623024
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5750923, 24.5753403
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6480713, 32.6470108
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7940140, 27.7926559
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4717789, 34.4728622
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8076019, 31.8049850
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0606079, 20.0584717
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5593987, 19.5574074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1732

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8914528, upper bound: 17.8810998
time: 23.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8928181, upper bound: 17.8797438
time: 26.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0043488, 19.0063477
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7922440, 31.7939987
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4040527, 21.4037933
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6665421, 34.6675873
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8893852, 33.8897095
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1773911, 36.1767883
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8818436, 25.8769913
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3932991, 34.3936653
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4692917, 24.4703712
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3630905, 33.3629646
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4813080, 52.4825134
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4192429, 24.4199829
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5483131, 21.5484352
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7415886, 19.7425919
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3947525, 31.3949356
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3801727, 22.3799477
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5220642, 21.5224495
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3763123, 21.3770294
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7262039, 37.7271042
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5309639, 24.5331993
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0144653, 24.0144310
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243713, 35.8249054
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7928467, 25.7941742
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1400185, 25.1403828
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3108521, 22.3089027
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2683945, 32.2649574
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8409958, 27.8399506
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2648010, 26.2611923
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5778999, 24.5738297
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6470718, 32.6452904
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7971802, 27.7928963
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4760437, 34.4698029
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8002739, 31.7999954
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0581169, 20.0556965
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5538750, 19.5540733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 707

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8788554, upper bound: 17.8900611
time: 20.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8768024, upper bound: 17.8921177
time: 15.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0042496, 19.0064468
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7923431, 31.7939034
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4040070, 21.4038353
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6665573, 34.6675720
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8894081, 33.8896942
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1773758, 36.1768074
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8817978, 25.8770370
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933220, 34.3936424
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4693756, 24.4702950
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3630524, 33.3629990
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4813385, 52.4824905
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4192581, 24.4199657
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5482750, 21.5484734
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7415810, 19.7425957
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3947830, 31.3949127
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3802032, 22.3799133
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5220795, 21.5224380
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3763962, 21.3769493
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7262115, 37.7270889
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5310249, 24.5331421
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0144424, 24.0144501
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243866, 35.8248901
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7929306, 25.7940903
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1400414, 25.1403618
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3108368, 22.3089256
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2682800, 32.2650757
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8409805, 27.8399658
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2647095, 26.2612762
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5778465, 24.5738907
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6469650, 32.6453972
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7970734, 27.7930126
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4759674, 34.4698715
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8006248, 31.7996445
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0580864, 20.0557251
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5540237, 19.5539265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8777123, upper bound: 17.8906357
time: 19.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8790059, upper bound: 17.8893421
time: 24.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0041122, 19.0066490
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7923126, 31.7945557
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4040527, 21.4039249
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6665573, 34.6676674
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8895874, 33.8895493
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1772614, 36.1770363
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8815308, 25.8773651
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933029, 34.3936119
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4695663, 24.4705734
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3629913, 33.3632126
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4812164, 52.4826660
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4193764, 24.4200001
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5486031, 21.5484467
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7417107, 19.7425880
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3949203, 31.3948898
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3802490, 22.3799973
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5224609, 21.5220795
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3766479, 21.3770294
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7264404, 37.7269135
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5314560, 24.5329895
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0145416, 24.0144577
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243408, 35.8249435
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7929459, 25.7942276
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1401787, 25.1402950
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3107796, 22.3090553
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2683411, 32.2651367
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8411484, 27.8398666
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2651749, 26.2612152
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5780983, 24.5738144
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6474876, 32.6454086
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7974472, 27.7929802
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4761429, 34.4699249
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8005371, 31.8013000
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0580788, 20.0557594
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5536423, 19.5546799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1426

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 567

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8691844, upper bound: 17.8827932
time: 21.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8566652, upper bound: 17.8953001
time: 16.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0039902, 19.0066872
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7922058, 31.7945938
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.4040680, 21.4039860
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6664734, 34.6676941
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8895721, 33.8895798
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1772614, 36.1770515
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8815994, 25.8772926
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3932190, 34.3936958
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4695663, 24.4705772
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3629608, 33.3632126
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4812164, 52.4826965
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4193878, 24.4200115
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5486336, 21.5483780
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7417412, 19.7424927
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3949509, 31.3947525
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3801422, 22.3800621
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5224609, 21.5221176
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3766632, 21.3769302
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7264404, 37.7269135
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5314560, 24.5330238
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0145569, 24.0144348
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8243408, 35.8249435
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7930069, 25.7942619
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1402245, 25.1402092
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3107643, 22.3090630
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2683258, 32.2651443
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8410645, 27.8399086
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2652206, 26.2611008
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5781364, 24.5737419
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6472969, 32.6455460
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7974777, 27.7929630
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4761734, 34.4699554
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8002853, 31.8014793
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0579033, 20.0558548
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5534248, 19.5548401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1357

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 730

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8709197, upper bound: 17.8963873
time: 19.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8702595, upper bound: 17.8970467
time: 25.12 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 47.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8677400, upper bound: 17.8875337
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8677193, upper bound: 17.8875444
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8745342, upper bound: 17.8956285
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8758978, upper bound: 17.8942667
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8548817, upper bound: 17.8817380
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8886062, upper bound: 17.8480336
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8984455, upper bound: 17.8850003
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.9002825, upper bound: 17.8831629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8981351, upper bound: 17.8854691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8981995, upper bound: 17.8854035
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8978835, upper bound: 17.8834855
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8978212, upper bound: 17.8835724
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8549751, upper bound: 17.8847328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8549751, upper bound: 17.8509735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8943498, upper bound: 17.8791982
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8932696, upper bound: 17.8802841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8914528, upper bound: 17.8810998
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8928181, upper bound: 17.8797438
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8788554, upper bound: 17.8900611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8768024, upper bound: 17.8921177
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8777123, upper bound: 17.8906357
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8790059, upper bound: 17.8893421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8691844, upper bound: 17.8827932
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8566652, upper bound: 17.8953001
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8709197, upper bound: 17.8963873
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 47.18
Output dim: 10, lower bound: -17.8702595, upper bound: 17.8970467

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0195503, 19.0241013
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7899017, 31.7908478
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3641281, 21.3676929
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6549225, 34.6568871
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8794060, 33.8774185
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1708069, 36.1748276
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8390617, 25.8350716
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3907433, 34.3887520
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4763985, 24.4727402
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3542252, 33.3623009
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4660416, 52.4660721
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4066048, 24.4039440
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5293884, 21.5250778
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7260208, 19.7254639
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3690186, 31.3645935
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3706665, 22.3702316
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4861069, 21.4788933
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3676529, 21.3651619
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7091293, 37.7057419
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4940109, 24.4902229
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9949036, 23.9935036
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8031311, 35.7997818
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7772141, 25.7750435
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1159744, 25.1099205
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3208160, 22.3250160
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2469788, 32.2462387
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8340950, 27.8358269
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2575493, 26.2535362
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5735664, 24.5702972
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6101837, 32.6132736
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7848434, 27.7833920
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4519958, 34.4439850
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7798843, 31.7824287
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0569038, 20.0593300
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5782318, 19.5915642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 513

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1339

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8664368, upper bound: 17.8860096
time: 15.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8662157, upper bound: 17.8862318
time: 25.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0210457, 19.0226002
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7944946, 31.7862625
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3577118, 21.3741074
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6575851, 34.6542244
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8813744, 33.8754425
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1679611, 36.1776772
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8242912, 25.8498344
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3909721, 34.3885307
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4767952, 24.4723396
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3531418, 33.3633842
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4670944, 52.4650269
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4095306, 24.4010181
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5295563, 21.5249138
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7274933, 19.7239876
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3700180, 31.3635941
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3745041, 22.3663940
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4887848, 21.4762115
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3682785, 21.3645325
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7119904, 37.7028809
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4995880, 24.4846420
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9974976, 23.9909096
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8059235, 35.7969894
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7794266, 25.7728386
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1162415, 25.1096535
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3206100, 22.3252182
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2399597, 32.2532578
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8364449, 27.8334885
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2509422, 26.2601433
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5678062, 24.5760498
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6062393, 32.6172180
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7831955, 27.7850304
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4381485, 34.4578362
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7776260, 31.7846909
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0555611, 20.0606747
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5846634, 19.5851326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 519

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8654451, upper bound: 17.8698854
time: 27.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8500569, upper bound: 17.8852824
time: 30.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0190277, 19.0220604
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7949677, 31.7896576
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3480606, 21.3612900
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6526031, 34.6525688
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8789902, 33.8738480
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1685257, 36.1785469
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8252106, 25.8414001
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3892555, 34.3875237
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4763908, 24.4706993
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3496552, 33.3605194
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4662857, 52.4661942
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4042435, 24.3957767
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5251083, 21.5187263
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7252579, 19.7220612
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3686523, 31.3613434
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3729706, 22.3678818
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4805222, 21.4668617
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3639984, 21.3591461
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7044220, 37.6954880
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4922943, 24.4790115
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9945831, 23.9890785
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8022156, 35.7944641
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7742958, 25.7674789
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1110153, 25.1025562
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3197365, 22.3271065
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2445602, 32.2527542
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8341293, 27.8335266
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2568817, 26.2606888
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5738678, 24.5769501
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6112823, 32.6185112
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7867393, 27.7873650
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4516602, 34.4595909
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7798882, 31.7848625
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0565758, 20.0605125
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5823860, 19.5915070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8744166, upper bound: 17.8901424
time: 26.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8690534, upper bound: 17.8955109
time: 26.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0189896, 19.0220985
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7931824, 31.7914467
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3515244, 21.3578262
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6533737, 34.6517982
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8776932, 33.8751411
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1718674, 36.1752052
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8308640, 25.8357468
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3896294, 34.3871460
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4746819, 24.4724121
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3514175, 33.3587532
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4673538, 52.4651184
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4009705, 24.3990517
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5229874, 21.5208549
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7239227, 19.7233963
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3669510, 31.3630447
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3720398, 22.3688049
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4762497, 21.4711266
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3619995, 21.3611412
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7012939, 37.6986160
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4879913, 24.4833107
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9929047, 23.9907608
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8004379, 35.7962494
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7719307, 25.7698441
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1085358, 25.1050377
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3227119, 22.3241310
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2464676, 32.2508430
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8341064, 27.8335419
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2578659, 26.2597084
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5743713, 24.5764465
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6113968, 32.6183929
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7867088, 27.7873917
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4536667, 34.4575806
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7799416, 31.7848091
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0567207, 20.0603619
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5842552, 19.5896378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1374

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8728804, upper bound: 17.8912621
time: 26.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8728812, upper bound: 17.8912611
time: 22.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9729328, 18.9825897
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7943268, 31.7910194
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3195267, 21.3126488
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6230011, 34.6156540
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8722687, 33.8706474
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2026482, 36.2008743
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8053131, 25.8035774
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3416100, 34.3304901
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4697266, 24.4658699
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3401337, 33.3357162
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4322052, 52.4259567
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3657036, 24.3702946
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5049438, 21.5085030
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7083817, 19.7120056
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3638229, 31.3671188
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3683128, 22.3648567
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.4419022, 21.4469604
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3660736, 21.3637619
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6842422, 37.6853333
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.4471817, 24.4529114
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9823990, 23.9852219
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7988815, 35.7996521
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7642441, 25.7665520
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0931320, 25.0962944
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3194962, 22.3224068
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2327957, 32.2359352
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8045883, 27.8098564
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2420273, 26.2464485
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5411453, 24.5487747
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5813446, 32.5820770
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7200165, 27.7320004
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4363556, 34.4407463
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7521896, 31.7549057
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0110893, 20.0179672
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5955372, 19.5887718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 519

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1423

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8883534, upper bound: 17.8466587
time: 17.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8872343, upper bound: 17.8477765
time: 15.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0086861, 19.0087852
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7928925, 31.7938652
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3926277, 21.3865395
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6671143, 34.6657143
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8839111, 33.8864937
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1732025, 36.1690979
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8734589, 25.8666458
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933563, 34.3902664
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4692612, 24.4699478
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3540726, 33.3500938
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4779053, 52.4742584
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4112473, 24.4156723
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5402832, 21.5431404
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7399940, 19.7405586
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3920517, 31.3917770
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3666573, 22.3710365
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5050011, 21.5102654
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3739319, 21.3720169
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7156219, 37.7191620
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5177574, 24.5236893
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0079803, 24.0097427
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8199081, 35.8205185
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7932625, 25.7909927
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306534, 25.1333351
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3113518, 22.3125267
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2675629, 32.2697067
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8323784, 27.8373642
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2635956, 26.2664223
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5762291, 24.5792503
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6441956, 32.6471710
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7848282, 27.7918987
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4737854, 34.4768219
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8022614, 31.8046303
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0512428, 20.0560989
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5582352, 19.5581837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1357

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1374

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8954440, upper bound: 17.8819964
time: 21.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8954450, upper bound: 17.8819955
time: 19.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0087547, 19.0087166
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7928925, 31.7938614
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3926506, 21.3865166
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6672440, 34.6655807
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8838348, 33.8865623
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1732788, 36.1690216
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8734589, 25.8666496
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3929901, 34.3906288
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4692307, 24.4699783
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3541718, 33.3499947
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4776764, 52.4744949
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4111710, 24.4157505
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5402222, 21.5431976
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7400246, 19.7405357
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3919983, 31.3918381
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3665962, 22.3710938
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5049706, 21.5102959
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3737640, 21.3721848
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7155838, 37.7191925
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5177116, 24.5237389
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0079727, 24.0097504
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8197784, 35.8206558
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7932625, 25.7909889
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1306381, 25.1333580
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3114738, 22.3124046
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2678452, 32.2694321
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8323479, 27.8373947
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2634659, 26.2665520
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5761528, 24.5793228
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6447372, 32.6466370
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7848206, 27.7919102
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4737167, 34.4768867
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8027496, 31.8041458
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0515404, 20.0558014
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5582466, 19.5581741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1407

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 707

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8997944, upper bound: 17.8806241
time: 25.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8977431, upper bound: 17.8826756
time: 30.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -19.0088501, 19.0086727
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7923965, 31.7936554
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.3926430, 21.3866043
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.6677322, 34.6666489
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8838196, 33.8865318
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1735764, 36.1691628
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.8727264, 25.8659286
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.3933487, 34.3911514
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4686432, 24.4701881
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3545990, 33.3504677
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4779663, 52.4753189
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.4114990, 24.4157238
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.5403900, 21.5430946
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.7399940, 19.7407112
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3916855, 31.3919830
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3668976, 22.3708267
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.5047874, 21.5102119
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3733444, 21.3729324
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.7155914, 37.7192535
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.5179329, 24.5238228
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -24.0080338, 24.0096893
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.8196640, 35.8209000
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7922897, 25.7913818
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.1304626, 25.1333599
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3126144, 22.3120461
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2687302, 32.2691879
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8333740, 27.8370018
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2640305, 26.2655067
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5772209, 24.5788078
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.6462173, 32.6472054
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7860565, 27.7907009
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4747925, 34.4760933
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.8040390, 31.8048363
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0537739, 20.0559692
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.5579491, 19.5576706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8947838, upper bound: 17.8732494
time: 23.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8859188, upper bound: 17.8821028
time: 18.79 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 43.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8664368, upper bound: 17.8860096
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8662157, upper bound: 17.8862318
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8654451, upper bound: 17.8698854
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8500569, upper bound: 17.8852824
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8744166, upper bound: 17.8901424
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8690534, upper bound: 17.8955109
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8728804, upper bound: 17.8912621
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8728812, upper bound: 17.8912611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8883534, upper bound: 17.8466587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8872343, upper bound: 17.8477765
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8954440, upper bound: 17.8819964
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8954450, upper bound: 17.8819955
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8997944, upper bound: 17.8806241
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8977431, upper bound: 17.8826756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8947838, upper bound: 17.8732494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 43.96
Output dim: 10, lower bound: -17.8859188, upper bound: 17.8821028
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8981995, upper bound: 17.8854035
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8978835, upper bound: 17.8834855
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8978212, upper bound: 17.8835724
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8549751, upper bound: 17.8847328
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8943498, upper bound: 17.8791982
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8932696, upper bound: 17.8802841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8914528, upper bound: 17.8810998
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8928181, upper bound: 17.8797438
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8788554, upper bound: 17.8900611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8768024, upper bound: 17.8921177
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8777123, upper bound: 17.8906357
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8790059, upper bound: 17.8893421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8566652, upper bound: 17.8953001
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8709197, upper bound: 17.8963873
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 43.96
Output dim: 10, lower bound: -17.8702595, upper bound: 17.8970467

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 32.26 + 1782.68 = 1814.94 seconds
