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
execution time: IAR + RelationalAnalysis = 2.79 + 29.86 = 32.65 seconds
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
time: 30.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9013316, upper bound: 17.8879850
time: 20.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 51.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 51.52
Output dim: 10, lower bound: -17.8879850, upper bound: 17.9013316
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 51.52
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8533434, upper bound: 17.9003891
time: 16.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8870448, upper bound: 17.8666718
time: 20.42 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

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
time: 20.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.9003891, upper bound: 17.8533434
time: 17.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 40.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 40.12
Output dim: 10, lower bound: -17.8533434, upper bound: 17.9003891
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 40.12
Output dim: 10, lower bound: -17.8870448, upper bound: 17.8666718
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 40.12
Output dim: 10, lower bound: -17.8666718, upper bound: 17.8870448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 40.12
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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
time: 25.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8748835
time: 25.31 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8661380
time: 29.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8865117, upper bound: 17.8411438
time: 23.73 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8865117
time: 30.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8615749
time: 23.39 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

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
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8528097
time: 25.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8998561, upper bound: 17.8278349
time: 20.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 48.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8998561
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8748835
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8661380
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8865117, upper bound: 17.8411438
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8278349, upper bound: 17.8865117
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8528097, upper bound: 17.8615749
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 48.06
Output dim: 10, lower bound: -17.8615749, upper bound: 17.8528097
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 48.06
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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
time: 18.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8276682, upper bound: 17.8984279
time: 17.20 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8849826, upper bound: 17.8409809
time: 31.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8863416, upper bound: 17.8397133
time: 20.06 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

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
time: 18.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8849826
time: 20.13 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1640

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8984279, upper bound: 17.8276682
time: 19.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8262730
time: 32.19 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 54.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8262730, upper bound: 17.8996847
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8276682, upper bound: 17.8984279
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8849826, upper bound: 17.8409809
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8863416, upper bound: 17.8397133
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8397133, upper bound: 17.8863416
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8409809, upper bound: 17.8849826
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 54.67
Output dim: 10, lower bound: -17.8984279, upper bound: 17.8276682
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 54.67
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8123254, upper bound: 17.8988680
time: 23.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8256596, upper bound: 17.8789810
time: 27.36 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8976116
time: 24.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8270518, upper bound: 17.8777205
time: 29.69 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8642743, upper bound: 17.8403622
time: 28.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8841707, upper bound: 17.8270277
time: 20.69 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8390951
time: 20.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8855296, upper bound: 17.8257550
time: 31.97 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8257550, upper bound: 17.8855296
time: 23.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8390950, upper bound: 17.8656474
time: 19.34 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8841707
time: 26.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8270277, upper bound: 17.8642743
time: 18.06 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8777204, upper bound: 17.8270518
time: 23.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8976116, upper bound: 17.8137332
time: 18.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 44.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8123254, upper bound: 17.8988680
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8256596, upper bound: 17.8789810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8976116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8270518, upper bound: 17.8777205
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8642743, upper bound: 17.8403622
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8841707, upper bound: 17.8270277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8374811, upper bound: 17.8390951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8855296, upper bound: 17.8257550
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8257550, upper bound: 17.8855296
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8390950, upper bound: 17.8656474
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8137332, upper bound: 17.8841707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8270277, upper bound: 17.8642743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8777204, upper bound: 17.8270518
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 44.55
Output dim: 10, lower bound: -17.8976116, upper bound: 17.8137332

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.26 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8099304, upper bound: 17.8812193
time: 20.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7946628, upper bound: 17.8965503
time: 22.57 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1699

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8113450, upper bound: 17.8799615
time: 18.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7960758, upper bound: 17.8952888
time: 15.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9802666, 18.9881172
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8030701, 31.8061943
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2521286, 21.2308292
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5798111, 34.5655670
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8469810, 33.8518105
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2339249, 36.2181168
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7301712, 25.7051620
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2976379, 34.2784576
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4814606, 24.4827576
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3194962, 33.3117294
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4615784, 52.4427643
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3049812, 24.3243675
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4541092, 21.4656487
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6758652, 19.6823311
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3350830, 31.3372269
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3545990, 22.3583031
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3553925, 21.3757591
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3527908, 21.3541107
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6301727, 37.6437073
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3484268, 24.3706322
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9520187, 23.9581757
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7775726, 35.7793884
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7320557, 25.7362976
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0483780, 25.0587502
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3514709, 22.3486862
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2196426, 32.2224121
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8252335, 27.8374977
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2425919, 26.2462082
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5211945, 24.5288010
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5476608, 32.5580750
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7126389, 27.7345543
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4287415, 34.4335098
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7375069, 31.7463608
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9922981, 20.0044231
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6454964, 19.6419983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8832240, upper bound: 17.8080987
time: 33.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8678755, upper bound: 17.8233495
time: 22.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9881172, 18.9802685
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8061981, 31.8030701
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2308311, 21.2521286
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5655670, 34.5798111
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8518181, 33.8469810
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2181168, 36.2339249
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7051620, 25.7301712
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2784576, 34.2976303
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4827576, 24.4814568
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3117294, 33.3194962
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4427643, 52.4615784
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3243675, 24.3049793
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4656448, 21.4541054
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6823349, 19.6758690
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3372192, 31.3350830
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3583069, 22.3545990
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3757629, 21.3553886
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3541107, 21.3527870
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6437073, 37.6301727
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3706284, 24.3484306
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9581680, 23.9520226
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7793884, 35.7775803
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7362976, 25.7320557
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0587540, 25.0483742
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3486938, 22.3514748
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2224121, 32.2196426
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8375015, 27.8252335
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2462082, 26.2425880
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5288010, 24.5211983
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5580750, 32.5476570
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7345505, 27.7126427
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4335098, 34.4287415
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7463646, 31.7375031
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0044250, 19.9922981
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6419983, 19.6454964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8233495, upper bound: 17.8678755
time: 18.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8080987, upper bound: 17.8832240
time: 19.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9777946, 18.9905930
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8043289, 31.8049316
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2513924, 21.2315655
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5798645, 34.5655136
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8482628, 33.8505287
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2321243, 36.2199173
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7283936, 25.7069397
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2982712, 34.2778244
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4831314, 24.4810829
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3197708, 33.3114586
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4610596, 52.4432907
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3058510, 24.3234940
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4554749, 21.4642830
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6754227, 19.6827812
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3352203, 31.3370819
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3549805, 22.3579178
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3574829, 21.3736687
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3537521, 21.3531418
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6323700, 37.6415100
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3488235, 24.3702354
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9523849, 23.9578094
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7779694, 35.7789993
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7323074, 25.7360458
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0497055, 25.0574207
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3493347, 22.3508263
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2189484, 32.2231064
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8245163, 27.8382111
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2421646, 26.2466278
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5200119, 24.5299835
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5497513, 32.5559845
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7119217, 27.7352753
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4286270, 34.4336243
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7377129, 31.7461548
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9918900, 20.0048332
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6451797, 19.6423149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8952888, upper bound: 17.7960758
time: 19.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8799615, upper bound: 17.8113450
time: 19.17 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 41.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8099304, upper bound: 17.8812193
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.7946628, upper bound: 17.8965503
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8113450, upper bound: 17.8799615
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.7960758, upper bound: 17.8952888
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8832240, upper bound: 17.8080987
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8678755, upper bound: 17.8233495
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8233495, upper bound: 17.8678755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8080987, upper bound: 17.8832240
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8952888, upper bound: 17.7960758
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 41.55
Output dim: 10, lower bound: -17.8799615, upper bound: 17.8113450

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9856796, 18.9812660
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8035965, 31.8021927
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2264633, 21.2554970
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5645142, 34.5791321
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8524590, 33.8463364
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2157898, 36.2352104
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6983795, 25.7231369
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2782021, 34.2972755
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4830818, 24.4810867
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3059311, 33.3236694
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4437256, 52.4604568
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3229904, 24.3017979
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4675827, 21.4520798
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6830139, 19.6749458
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3408508, 31.3305588
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3582344, 22.3544464
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3791962, 21.3514709
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3532486, 21.3512459
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6450424, 37.6293411
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3709869, 24.3441505
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9602737, 23.9496422
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7816391, 35.7740097
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7383461, 25.7293396
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0615692, 25.0453930
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3477783, 22.3523560
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2172775, 32.2188759
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8367157, 27.8244629
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2430038, 26.2388077
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5261536, 24.5169983
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5524597, 32.5523605
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7322998, 27.7092590
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4264221, 34.4234695
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7426758, 31.7421265
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0016651, 19.9938965
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6333580, 19.6509495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1591

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7785345, upper bound: 17.8952654
time: 28.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7933776, upper bound: 17.8804614
time: 30.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9886284, 18.9783173
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8022232, 31.8035736
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2312622, 21.2506981
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5645065, 34.5791397
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8505287, 33.8482666
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2194672, 36.2315369
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7022552, 25.7192650
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2774391, 34.2980423
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4809914, 24.4831772
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3106384, 33.3189659
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4432068, 52.4609909
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3208885, 24.3040543
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4642715, 21.4554405
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6822510, 19.6757469
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3366699, 31.3347397
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3577919, 22.3548927
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3734436, 21.3572159
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3518906, 21.3526001
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6414185, 37.6329041
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3672485, 24.3476906
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9577103, 23.9522133
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7781143, 35.7775269
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7353249, 25.7323608
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0573349, 25.0496235
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3508148, 22.3493195
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2210617, 32.2150955
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8375702, 27.8236122
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2442551, 26.2375526
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5276566, 24.5154915
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5556488, 32.5491714
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7331696, 27.7083912
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4293594, 34.4205360
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7459412, 31.7388611
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -20.0042515, 19.9913120
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6402740, 19.6443138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1591

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7799627, upper bound: 17.8940033
time: 28.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7947913, upper bound: 17.8792008
time: 27.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9783173, 18.9886284
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8035812, 31.8022156
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2506981, 21.2312641
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5791397, 34.5645065
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8482704, 33.8505249
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2315369, 36.2194672
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7192688, 25.7022514
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2980385, 34.2774391
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4831734, 24.4809952
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3189697, 33.3106384
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4609985, 52.4431992
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3040543, 24.3208885
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4554367, 21.4642754
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6757507, 19.6822548
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3347397, 31.3366699
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3548927, 22.3577919
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3572235, 21.3734474
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3526001, 21.3518944
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6329041, 37.6414185
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3476868, 24.3672447
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9522171, 23.9577065
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7775269, 35.7781219
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7323647, 25.7353210
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0496292, 25.0573387
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3493195, 22.3508110
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2150955, 32.2210579
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8236084, 27.8375664
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2375565, 26.2442589
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5154877, 24.5276566
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5491714, 32.5556488
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7083893, 27.7331696
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4205322, 34.4293594
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7388611, 31.7459373
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9913120, 20.0042477
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6443138, 19.6402740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1591

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8439640, upper bound: 17.7947913
time: 18.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8940033, upper bound: 17.7799627
time: 20.42 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 41.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 41.08
Output dim: 10, lower bound: -17.7785345, upper bound: 17.8952654
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 41.08
Output dim: 10, lower bound: -17.7933776, upper bound: 17.8804614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 41.08
Output dim: 10, lower bound: -17.7799627, upper bound: 17.8940033
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 41.08
Output dim: 10, lower bound: -17.7947913, upper bound: 17.8792008
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 41.08
Output dim: 10, lower bound: -17.8439640, upper bound: 17.7947913
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 41.08
Output dim: 10, lower bound: -17.8940033, upper bound: 17.7799627

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9743462, 18.9817162
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8031082, 31.8004646
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2202301, 21.2515602
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5652466, 34.5789566
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8474884, 33.8382874
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2048111, 36.2282143
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6911316, 25.7177544
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2646141, 34.2751312
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4845924, 24.4782028
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3022614, 33.3236771
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4356995, 52.4474869
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3179626, 24.2942791
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4595108, 21.4390488
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6829300, 19.6750717
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3339005, 31.3193054
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3526077, 22.3453369
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3665924, 21.3310242
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3479462, 21.3426781
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6348877, 37.6136856
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3655777, 24.3357964
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9524841, 23.9370537
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7744675, 35.7624130
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7357635, 25.7253189
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0541992, 25.0334797
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3420105, 22.3537636
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2063293, 32.2119980
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8305206, 27.8205872
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2428207, 26.2386665
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5211639, 24.5143280
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5413895, 32.5454407
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7226334, 27.7031517
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4237518, 34.4215965
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7255173, 31.7315140
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9966755, 19.9947395
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6253014, 19.6526375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7748445, upper bound: 17.8771763
time: 21.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7604044, upper bound: 17.8915216
time: 23.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9772949, 18.9787674
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8017273, 31.8018494
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2250290, 21.2467594
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5652390, 34.5789642
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8455582, 33.8402214
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2084885, 36.2245407
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6950073, 25.7138863
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2638512, 34.2758942
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4825020, 24.4802895
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3069611, 33.3189697
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4351654, 52.4480133
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3158646, 24.2965355
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4562149, 21.4424095
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6821747, 19.6758766
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3297195, 31.3234940
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3521652, 22.3457870
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3608398, 21.3367691
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3465958, 21.3440285
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6312637, 37.6172562
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3618546, 24.3393364
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9499207, 23.9396248
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7709427, 35.7659378
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7327423, 25.7283401
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0499649, 25.0377102
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3450470, 22.3507309
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2101059, 32.2082214
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8313751, 27.8197403
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2440720, 26.2374115
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5226669, 24.5128250
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5445786, 32.5422554
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7235031, 27.7022858
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4266815, 34.4186630
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7287827, 31.7282486
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9992619, 19.9921570
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6322174, 19.6459999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7762609, upper bound: 17.8759092
time: 19.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7618373, upper bound: 17.8902595
time: 23.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9787674, 18.9772949
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8018494, 31.8017235
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2467575, 21.2250290
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5789642, 34.5652428
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8402176, 33.8455505
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2245407, 36.2084885
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7138824, 25.6949997
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2758980, 34.2638512
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4802895, 24.4825058
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3189697, 33.3069649
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4480133, 52.4351730
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.2965355, 24.3158627
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4424057, 21.4562111
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6758728, 19.6821747
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3234940, 31.3297195
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3457870, 22.3521576
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3367615, 21.3608475
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3440247, 21.3465958
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6172562, 37.6312637
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3393326, 24.3618507
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9396210, 23.9499168
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7659378, 35.7709427
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7283401, 25.7327423
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0377121, 25.0499668
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3507309, 22.3450470
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2082214, 32.2101059
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8197403, 27.8313713
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2374191, 26.2440758
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5128250, 24.5226707
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5422592, 32.5445786
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7022858, 27.7235050
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4186630, 34.4266853
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7282486, 31.7287827
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9921589, 19.9992599
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6459999, 19.6322174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8902595, upper bound: 17.7618373
time: 19.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8759092, upper bound: 17.7762610
time: 17.06 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 38.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 38.90
Output dim: 10, lower bound: -17.7748445, upper bound: 17.8771763
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 38.90
Output dim: 10, lower bound: -17.7604044, upper bound: 17.8915216
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 38.90
Output dim: 10, lower bound: -17.7762609, upper bound: 17.8759092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 38.90
Output dim: 10, lower bound: -17.7618373, upper bound: 17.8902595
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 38.90
Output dim: 10, lower bound: -17.8902595, upper bound: 17.7618373
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 38.90
Output dim: 10, lower bound: -17.8759092, upper bound: 17.7762610

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9731598, 18.9773560
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8000183, 31.7991104
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2176819, 21.2509460
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5590057, 34.5773392
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8451080, 33.8372421
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2047272, 36.2283478
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6904144, 25.7176437
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2564316, 34.2734985
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4830513, 24.4769974
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.2986298, 33.3227768
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4284744, 52.4457321
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3178825, 24.2941418
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4593811, 21.4385147
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6820679, 19.6722984
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3331604, 31.3174744
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3525238, 22.3452950
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3665886, 21.3307037
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3479004, 21.3434372
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6348267, 37.6131744
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3648987, 24.3329201
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9524765, 23.9370270
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7738800, 35.7631836
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7356415, 25.7273979
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0538330, 25.0319710
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3402252, 22.3473587
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2042465, 32.2038116
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8290024, 27.8142815
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2410812, 26.2316093
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5187225, 24.5044022
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5407410, 32.5427475
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7194748, 27.6900196
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4210739, 34.4110603
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7250443, 31.7295494
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9954853, 19.9898014
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6252937, 19.6526184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7490412, upper bound: 17.8906206
time: 16.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7594993, upper bound: 17.8804595
time: 25.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9761124, 18.9744091
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7986374, 31.8004913
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2224808, 21.2461452
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5589905, 34.5773506
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8431702, 33.8391762
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2084045, 36.2246742
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6942902, 25.7137680
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2556686, 34.2742653
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4809608, 24.4790878
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3033295, 33.3180733
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4279404, 52.4462662
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3157845, 24.2963982
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4560776, 21.4418755
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6813126, 19.6731033
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3289719, 31.3216629
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3520737, 22.3457413
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3608360, 21.3364487
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3465500, 21.3447914
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6312027, 37.6167450
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3611603, 24.3364601
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9499130, 23.9396019
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7703629, 35.7667084
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7326202, 25.7304153
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0495987, 25.0362015
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3432617, 22.3443222
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2080231, 32.2000275
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8298569, 27.8134346
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2423325, 26.2303581
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5202255, 24.5028992
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5439301, 32.5395622
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7203445, 27.6891537
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4240036, 34.4081268
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7283096, 31.7262878
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9980717, 19.9872189
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6322136, 19.6459846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7504774, upper bound: 17.8893580
time: 24.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7609319, upper bound: 17.8791964
time: 20.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9744110, 18.9761124
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8004913, 31.7986374
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2461472, 21.2224808
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5773468, 34.5589943
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8391800, 33.8431702
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2246704, 36.2084007
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.7137604, 25.6942863
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2742691, 34.2556686
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4790840, 24.4809647
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3180695, 33.3033333
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4462662, 52.4279404
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.2963982, 24.3157825
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4418716, 21.4560776
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6731033, 19.6813126
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3216629, 31.3289719
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3457413, 22.3520737
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3364525, 21.3608360
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3447952, 21.3465500
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6167450, 37.6312027
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3364563, 24.3611603
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9395981, 23.9499092
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7667007, 35.7703629
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7304077, 25.7326164
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0362015, 25.0496006
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3443146, 22.3432617
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2000275, 32.2080231
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8134384, 27.8298531
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2303543, 26.2423325
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5028992, 24.5202255
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5395584, 32.5439301
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.6891556, 27.7203465
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4081268, 34.4240036
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7262878, 31.7283096
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9872227, 19.9980698
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6459846, 19.6322136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1756

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8791964, upper bound: 17.7609319
time: 21.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8893580, upper bound: 17.7504774
time: 20.31 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 43.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 43.74
Output dim: 10, lower bound: -17.7490412, upper bound: 17.8906206
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 43.74
Output dim: 10, lower bound: -17.7594993, upper bound: 17.8804595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 43.74
Output dim: 10, lower bound: -17.7504774, upper bound: 17.8893580
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 43.74
Output dim: 10, lower bound: -17.7609319, upper bound: 17.8791964
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 43.74
Output dim: 10, lower bound: -17.8791964, upper bound: 17.7609319
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 43.74
Output dim: 10, lower bound: -17.8893580, upper bound: 17.7504774

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9709740, 18.9764481
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.8001480, 31.7967758
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2005806, 21.2397270
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5578384, 34.5763779
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8433228, 33.8325081
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1950989, 36.2232971
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6674194, 25.7024460
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2551880, 34.2722588
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4841614, 24.4757538
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.2861557, 33.3143921
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4222183, 52.4408417
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3121834, 24.2842846
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4547348, 21.4313507
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6799850, 19.6696014
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3307419, 31.3143539
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3446045, 22.3331299
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3561172, 21.3146362
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3453293, 21.3400841
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6287918, 37.6030350
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3528900, 24.3149338
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9485703, 23.9313774
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7703247, 35.7576523
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7337875, 25.7254677
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0478249, 25.0230026
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3397827, 22.3484001
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2013779, 32.2020111
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8237038, 27.8064079
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2408066, 26.2312965
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5187225, 24.5043907
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5387001, 32.5407639
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7134171, 27.6811352
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4182358, 34.4087868
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7236481, 31.7282143
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9925537, 19.9858246
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6248779, 19.6524239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 732

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7389319, upper bound: 17.8896766
time: 19.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7477745, upper bound: 17.8794120
time: 26.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9739189, 18.9734993
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7987671, 31.7981567
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2053795, 21.2349281
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5578308, 34.5763855
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8413925, 33.8344421
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1987686, 36.2196236
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6712875, 25.6985741
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2544250, 34.2730255
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4820862, 24.4778442
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.2908630, 33.3096886
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4216690, 52.4413757
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3100853, 24.2865410
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4514389, 21.4347115
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6792297, 19.6704063
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3265533, 31.3185425
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3441620, 22.3335800
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3503799, 21.3203812
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3439713, 21.3414345
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6251678, 37.6066055
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3491592, 24.3184738
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9460068, 23.9339485
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7667999, 35.7611694
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7307663, 25.7284889
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0435829, 25.0272331
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3428192, 22.3453636
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.2051620, 32.1982307
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8245430, 27.8055573
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2420578, 26.2300415
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5202179, 24.5028877
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5418892, 32.5375786
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7142868, 27.6802692
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4211731, 34.4058533
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7269135, 31.7249527
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9951363, 19.9832420
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6317978, 19.6457882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 732

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.7403668, upper bound: 17.8884145
time: 26.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7492097, upper bound: 17.8781528
time: 23.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9734993, 18.9739227
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7981567, 31.7987671
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.2349281, 21.2053795
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5763779, 34.5578346
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8344421, 33.8413925
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.2196274, 36.1987724
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6985703, 25.6712875
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2730255, 34.2544250
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4778442, 24.4820786
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.3096924, 33.2908592
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4413834, 52.4216843
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.2865410, 24.3100872
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4347153, 21.4514351
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6704102, 19.6792297
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3185425, 31.3265610
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3335800, 22.3441582
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3203812, 21.3503723
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3414383, 21.3439751
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6066055, 37.6251678
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3184738, 24.3491554
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9339523, 23.9460030
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7611694, 35.7667999
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7284927, 25.7307625
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0272331, 25.0435886
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3453674, 22.3428192
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.1982346, 32.2051620
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8055611, 27.8245468
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2300339, 26.2420578
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5028839, 24.5202179
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5375786, 32.5418854
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.6802673, 27.7142868
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4058533, 34.4211693
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7249527, 31.7269096
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9832420, 19.9951363
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6457901, 19.6317940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 732

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8781528, upper bound: 17.7492097
time: 18.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8884145, upper bound: 17.7403668
time: 20.19 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 50.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 50.55
Output dim: 10, lower bound: -17.7389319, upper bound: 17.8896766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 50.55
Output dim: 10, lower bound: -17.7477745, upper bound: 17.8794120
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 50.55
Output dim: 10, lower bound: -17.7403668, upper bound: 17.8884145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 50.55
Output dim: 10, lower bound: -17.7492097, upper bound: 17.8781528
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 50.55
Output dim: 10, lower bound: -17.8781528, upper bound: 17.7492097
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 50.55
Output dim: 10, lower bound: -17.8884145, upper bound: 17.7403668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9571724, 18.9654598
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7846909, 31.7778511
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.1701736, 21.2160606
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5519409, 34.5708580
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8513870, 33.8378372
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1638107, 36.1970901
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6132355, 25.6575012
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2473450, 34.2638664
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.5004196, 24.4880142
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.2565002, 33.2915001
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4117126, 52.4309158
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3265953, 24.2949390
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4576225, 21.4310226
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6660767, 19.6529312
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3283691, 31.3082733
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3365860, 22.3224754
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3267250, 21.2792053
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3352509, 21.3293419
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6495895, 37.6192017
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3127136, 24.2664871
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9419098, 23.9227753
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7671356, 35.7510223
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7309990, 25.7224731
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0303497, 25.0005741
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3431320, 22.3522034
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.1951294, 32.1978607
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8213539, 27.8031311
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2398529, 26.2305908
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5175858, 24.5032539
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5254822, 32.5289955
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7105408, 27.6754284
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4114532, 34.4036446
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7242393, 31.7300377
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9878502, 19.9813900
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6136818, 19.6443710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 732
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
type: RSZ, layer: 1, pos: 732

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1790

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7277264, upper bound: 17.8697650
time: 14.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.7293461, upper bound: 17.8683093
time: 20.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.6287022, 5.8225431, -28.6287022, 5.8225431, -34.4512444, 34.4512444
1: -15.2265425, 11.1505833, -15.2265425, 11.1505833, -26.3771248, 26.3771248
2: -12.3355474, 10.9271336, -12.3355474, 10.9271336, -23.2626801, 23.2626801
3: -8.9950886, 15.8139057, -8.9950886, 15.8139057, -24.8089943, 24.8089943
4: -12.7426891, 13.2859535, -12.7426891, 13.2859535, -26.0286427, 26.0286427
5: -9.9357424, 18.0674095, -9.9357424, 18.0674095, -28.0031509, 28.0031509
6: -27.4773502, -2.9538975, -27.4773502, -2.9538975, -18.9601250, 18.9625130
7: -13.2555027, 17.7249088, -13.2555027, 17.7249088, -30.9804115, 30.9804115
8: -16.9944477, 15.8259163, -16.9944477, 15.8259163, -31.7833176, 31.7792320
9: -12.2571983, 13.5999184, -12.2571983, 13.5999184, -21.1749725, 21.2112617
10: -13.0781946, 24.7642326, -13.0781946, 24.7642326, -34.5519257, 34.5708656
11: -22.7050323, 12.8492756, -22.7050323, 12.8492756, -33.8494492, 33.8397675
12: -20.8690147, 15.4471054, -20.8690147, 15.4471054, -36.1674881, 36.1934128
13: -21.0863094, 11.3339386, -21.0863094, 11.3339386, -25.6171112, 25.6536331
14: -43.0412903, 3.4582219, -43.0412903, 3.4582219, -34.2465744, 34.2646332
15: -15.1078691, 9.8904247, -15.1078691, 9.8904247, -24.4983292, 24.4901009
16: -21.1429043, 13.1652775, -21.1429043, 13.1652775, -33.2612076, 33.2867966
17: -33.8622894, 27.5240059, -33.8622894, 27.5240059, -52.4111938, 52.4314499
18: -17.6785431, 7.9921055, -17.6785431, 7.9921055, -24.3244972, 24.2971954
19: -20.1052608, 2.0517590, -20.1052608, 2.0517590, -21.4543190, 21.4343796
20: -10.1728039, 10.3028851, -10.1728039, 10.3028851, -19.6653214, 19.6537323
21: -20.7056503, 7.2324162, -20.7056503, 7.2324162, -27.9380665, 27.9380665
22: -22.9295483, 9.3774834, -22.9295483, 9.3774834, -31.3241882, 31.3124619
23: -19.3672714, 4.2976866, -19.3672714, 4.2976866, -22.3361359, 22.3229256
24: -26.7588882, -1.6675973, -26.7588882, -1.6675973, -21.3209877, 21.2849503
25: -13.3050747, 9.5221395, -13.3050747, 9.5221395, -21.3338928, 21.3306961
26: -28.9389420, 8.8154926, -28.9389420, 8.8154926, -37.6459732, 37.6227646
27: -28.5978985, 0.3546729, -28.5978985, 0.3546729, -24.3089752, 24.2700233
28: -18.5431309, 6.3480358, -18.5431309, 6.3480358, -23.9393387, 23.9253502
29: -32.0854225, 5.0921278, -32.0854225, 5.0921278, -35.7636185, 35.7545395
30: -18.5028534, 8.4226856, -18.5028534, 8.4226856, -25.7279778, 25.7254944
31: -18.0225906, 8.5298271, -18.0225906, 8.5298271, -25.0261230, 25.0048046
32: -21.4231625, 4.2342806, -21.4231625, 4.2342806, -22.3461685, 22.3491669
33: -39.3339119, 1.1159987, -39.3339119, 1.1159987, -32.1989136, 32.1940842
34: -30.8308887, 2.1966033, -30.8308887, 2.1966033, -27.8222084, 27.8022842
35: -30.3201351, 2.4790554, -30.3201351, 2.4790554, -26.2411041, 26.2293396
36: -31.7673531, 0.2070944, -31.7673531, 0.2070944, -24.5190887, 24.5017509
37: -47.3777924, -6.5111041, -47.3777924, -6.5111041, -32.5286713, 32.5258064
38: -40.6754646, -2.1630316, -40.6754646, -2.1630316, -27.7114105, 27.6745605
39: -50.5764351, -5.9453330, -50.5764351, -5.9453330, -34.4143829, 34.4007111
40: -41.7258263, -3.3837671, -41.7258263, -3.3837671, -31.7275047, 31.7267723
41: -31.1759529, -4.2211390, -31.1759529, -4.2211390, -19.9904289, 19.9788055
42: -18.1341343, 2.5891733, -18.1341343, 2.5891733, -19.6205978, 19.6377373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=106, inp2_unstable=106, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 732
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
type: RSZ, layer: 1, pos: 732

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 32.65 + 1767.95 = 1800.60 seconds
