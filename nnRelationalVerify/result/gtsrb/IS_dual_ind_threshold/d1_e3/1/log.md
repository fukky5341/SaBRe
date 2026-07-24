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
execution time: IAR + RelationalAnalysis = 2.57 + 28.12 = 30.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -17.9025189, upper bound: 17.9025189

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1651

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8707439
time: 17.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8992939
time: 18.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 35.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 35.43
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8707439
IS_A2, status: Status.UNKNOWN, split count: 1, time: 35.43
Output dim: 10, lower bound: -17.8992941, upper bound: 17.8992939

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -28.6255093, 5.7930856, -28.6272793, 5.8094230, -34.4349327, 34.4203644
1: -15.2251434, 11.1352701, -15.2259121, 11.1437283, -26.3688717, 26.3611832
2: -12.3342991, 10.9022179, -12.3349762, 10.9160328, -23.2503319, 23.2371941
3: -8.9930210, 15.7852020, -8.9941616, 15.8011093, -24.7941303, 24.7793636
4: -12.7409334, 13.2668657, -12.7419138, 13.2774687, -26.0184021, 26.0087795
5: -9.9337006, 18.0416641, -9.9348230, 18.0559616, -27.9896622, 27.9764862
6: -27.4536705, -2.9562340, -27.4665203, -2.9549379, -18.9843788, 18.9979382
7: -13.2533560, 17.7111301, -13.2545185, 17.7187634, -30.9721184, 30.9656487
8: -16.9920235, 15.8024940, -16.9933662, 15.8154793, -31.7810974, 31.7687683
9: -12.2550011, 13.5712986, -12.2562017, 13.5872078, -21.3893280, 21.3744736
10: -13.0742912, 24.7216492, -13.0764370, 24.7453136, -34.6457672, 34.6241455
11: -22.6848392, 12.8449469, -22.6960201, 12.8473492, -33.8666115, 33.8751602
12: -20.8591709, 15.4395714, -20.8645515, 15.4437752, -36.1582718, 36.1650314
13: -21.0817719, 11.3025827, -21.0842648, 11.3199902, -25.8712540, 25.8555031
14: -43.0324860, 3.4210405, -43.0373383, 3.4416747, -34.3677711, 34.3521500
15: -15.1044292, 9.8577271, -15.1062946, 9.8758688, -24.4527893, 24.4360275
16: -21.1396294, 13.1514549, -21.1414185, 13.1591034, -33.3540802, 33.3488541
17: -33.8495483, 27.4984589, -33.8566284, 27.5126667, -52.4590912, 52.4518051
18: -17.6543846, 7.9889984, -17.6677837, 7.9907050, -24.3959465, 24.4070835
19: -20.0832596, 2.0506546, -20.0954781, 2.0512691, -21.5253372, 21.5368462
20: -10.1490479, 10.3012486, -10.1622200, 10.3021507, -19.7182617, 19.7307434
21: -20.6769218, 7.2297602, -20.6928787, 7.2312226, -27.9081440, 27.9226379
22: -22.9098930, 9.3754950, -22.9207954, 9.3765745, -31.3748245, 31.3840256
23: -19.3517685, 4.2955985, -19.3603783, 4.2967691, -22.3574638, 22.3677902
24: -26.7314415, -1.6693230, -26.7466755, -1.6683936, -21.4932213, 21.5072365
25: -13.2781763, 9.5205631, -13.2931147, 9.5214291, -21.3476410, 21.3623772
26: -28.9174175, 8.8135481, -28.9293404, 8.8146067, -37.7032089, 37.7140961
27: -28.5676422, 0.3525167, -28.5844345, 0.3537216, -24.5050125, 24.5207329
28: -18.5134926, 6.3459148, -18.5299606, 6.3470759, -23.9830322, 23.9984474
29: -32.0700150, 5.0893145, -32.0783501, 5.0908585, -35.8080063, 35.8149948
30: -18.4672394, 8.4194841, -18.4870148, 8.4212427, -25.7567902, 25.7753677
31: -17.9938583, 8.5273237, -18.0098000, 8.5286999, -25.1099815, 25.1245155
32: -21.4100571, 4.2306590, -21.4173241, 4.2326803, -22.2966499, 22.3034706
33: -39.3128166, 1.1133900, -39.3245392, 1.1148200, -32.2503204, 32.2607231
34: -30.8188400, 2.1944180, -30.8254986, 2.1956253, -27.8295135, 27.8349724
35: -30.3060951, 2.4776893, -30.3138962, 2.4784517, -26.2517014, 26.2586594
36: -31.7499657, 0.2059734, -31.7596302, 0.2065895, -24.5620308, 24.5706673
37: -47.3440094, -6.5138068, -47.3627243, -6.5122871, -32.6140060, 32.6322021
38: -40.6511612, -2.1652341, -40.6646576, -2.1640306, -27.7726364, 27.7853508
39: -50.5607529, -5.9465604, -50.5692596, -5.9458628, -34.4648132, 34.4718208
40: -41.7049942, -3.3854194, -41.7162857, -3.3844986, -31.7812195, 31.7943573
41: -31.1593208, -4.2238188, -31.1685181, -4.2223334, -20.0439987, 20.0521584
42: -18.1265030, 2.5865159, -18.1306648, 2.5879836, -19.5380478, 19.5479679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8530444
time: 21.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8530444
time: 20.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -28.6725674, 5.8214579, -28.6278000, 5.8201070, -34.4926758, 34.4492569
1: -15.2483616, 11.1516380, -15.2261744, 11.1492138, -26.3975754, 26.3778114
2: -12.3714943, 10.9271002, -12.3352833, 10.9253922, -23.2968864, 23.2623825
3: -9.0343380, 15.8122921, -8.9947300, 15.8114738, -24.8458118, 24.8070221
4: -12.7655144, 13.2862740, -12.7426014, 13.2842216, -26.0497360, 26.0288754
5: -9.9739227, 18.0668316, -9.9354591, 18.0656261, -28.0395489, 28.0022907
6: -27.4785366, -2.9274960, -27.4745636, -2.9541664, -19.0054646, 19.0417442
7: -13.2716341, 17.7262783, -13.2549019, 17.7237644, -30.9953995, 30.9811802
8: -17.0269775, 15.8250294, -16.9941483, 15.8238811, -31.8254089, 31.7909775
9: -12.2963686, 13.5996246, -12.2567177, 13.5979166, -21.4412155, 21.3987465
10: -13.1357470, 24.7664490, -13.0772200, 24.7615414, -34.7235565, 34.6636429
11: -22.7065201, 12.8714924, -22.7023983, 12.8486366, -33.8897896, 33.9093094
12: -20.8701324, 15.4551668, -20.8674545, 15.4466124, -36.1697311, 36.1974831
13: -21.1300430, 11.3363705, -21.0855942, 11.3315506, -25.9319916, 25.8863754
14: -43.0994301, 3.4585719, -43.0401535, 3.4556990, -34.4501534, 34.3827286
15: -15.1494083, 9.8889980, -15.1068630, 9.8872871, -24.5106468, 24.4673080
16: -21.1672459, 13.1667967, -21.1416130, 13.1639662, -33.3863754, 33.3619232
17: -33.9004974, 27.5224876, -33.8614197, 27.5216713, -52.5203705, 52.4754562
18: -17.6800613, 8.0239582, -17.6768456, 7.9914441, -24.4182053, 24.4521389
19: -20.1063519, 2.0845315, -20.1035309, 2.0514722, -21.5465240, 21.5792313
20: -10.1728945, 10.3402348, -10.1707821, 10.3025455, -19.7394066, 19.7785988
21: -20.7071857, 7.2748108, -20.7034149, 7.2320356, -27.9392204, 27.9782257
22: -22.9326115, 9.4038363, -22.9279137, 9.3765774, -31.3942795, 31.4194489
23: -19.3685951, 4.3201580, -19.3658752, 4.2974043, -22.3707733, 22.4074745
24: -26.7586708, -1.6229982, -26.7564526, -1.6681662, -21.5164833, 21.5633850
25: -13.3059673, 9.5624571, -13.3031311, 9.5213041, -21.3721848, 21.4173088
26: -28.9415245, 8.8420887, -28.9370842, 8.8149195, -37.7261581, 37.7513428
27: -28.5987911, 0.3990612, -28.5956573, 0.3543534, -24.5319443, 24.5794830
28: -18.5429420, 6.3883924, -18.5411949, 6.3475833, -24.0126038, 24.0528107
29: -32.0899277, 5.1121273, -32.0837135, 5.0916491, -35.8278732, 35.8446579
30: -18.5026798, 8.4698267, -18.4999657, 8.4222336, -25.7890396, 25.8392906
31: -18.0239925, 8.5692368, -18.0200691, 8.5291824, -25.1378860, 25.1765442
32: -21.4254761, 4.2503905, -21.4211063, 4.2340403, -22.3116188, 22.3357582
33: -39.3393173, 1.1508269, -39.3330574, 1.1151314, -32.2728882, 32.3087692
34: -30.8344193, 2.2143340, -30.8291740, 2.1960554, -27.8444443, 27.8572159
35: -30.3230209, 2.4986434, -30.3188019, 2.4786491, -26.2670593, 26.2844391
36: -31.7707634, 0.2305481, -31.7657280, 0.2069478, -24.5783310, 24.6013489
37: -47.3792114, -6.4696460, -47.3750763, -6.5115809, -32.6473541, 32.6898613
38: -40.6793137, -2.1402416, -40.6735992, -2.1634350, -27.7963562, 27.8214645
39: -50.5789795, -5.9272881, -50.5744934, -5.9457173, -34.4807358, 34.4975357
40: -41.7241364, -3.3578825, -41.7220535, -3.3842163, -31.8077774, 31.8420715
41: -31.1783237, -4.1956730, -31.1741905, -4.2213402, -20.0606728, 20.0856400
42: -18.1386204, 2.5993714, -18.1319618, 2.5887394, -19.5443535, 19.5895424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=106, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8815985
time: 17.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8980097
time: 16.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 36.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.27
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8530444
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.27
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8530444
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.27
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8815985
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.27
Output dim: 10, lower bound: -17.8974300, upper bound: 17.8980097

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -28.6231384, 5.7850070, -28.6220703, 5.7919869, -34.4151268, 34.4070778
1: -15.2242031, 11.1313725, -15.2238770, 11.1353016, -26.3595047, 26.3552494
2: -12.3333912, 10.8981295, -12.3330297, 10.9073782, -23.2407684, 23.2311592
3: -8.9850674, 15.7833290, -8.9768772, 15.7970581, -24.7821255, 24.7602062
4: -12.7403440, 13.2633057, -12.7406645, 13.2698069, -26.0101509, 26.0039711
5: -9.9270840, 18.0403671, -9.9204540, 18.0531235, -27.9802074, 27.9608212
6: -27.4331245, -2.9570308, -27.4218636, -2.9565678, -18.9676476, 18.9737663
7: -13.2496548, 17.7087669, -13.2464981, 17.7137794, -30.9634342, 30.9552650
8: -16.9910126, 15.7973785, -16.9911880, 15.8045559, -31.7720184, 31.7614822
9: -12.2470646, 13.5701618, -12.2389221, 13.5847349, -21.3783760, 21.3562965
10: -13.0602446, 24.7188530, -13.0458775, 24.7392483, -34.6248169, 34.5881958
11: -22.6776810, 12.8327875, -22.6804962, 12.8208838, -33.8326836, 33.8468666
12: -20.8454456, 15.4381514, -20.8349228, 15.4407063, -36.1412659, 36.1342583
13: -21.0669250, 11.3000031, -21.0525723, 11.3145313, -25.8510132, 25.8199005
14: -43.0290222, 3.4005456, -43.0299377, 3.3970623, -34.3227882, 34.3243942
15: -15.1038990, 9.8451099, -15.1051655, 9.8484573, -24.4349442, 24.4210968
16: -21.1291752, 13.1506853, -21.1188145, 13.1574230, -33.3344040, 33.3256035
17: -33.8471069, 27.4824390, -33.8513298, 27.4778309, -52.4218597, 52.4306107
18: -17.6528587, 7.9740000, -17.6645584, 7.9589567, -24.3623314, 24.3888206
19: -20.0799904, 2.0357614, -20.0883484, 2.0188091, -21.4895020, 21.5144539
20: -10.1459389, 10.2925978, -10.1555195, 10.2834835, -19.6949158, 19.7145042
21: -20.6730385, 7.2195635, -20.6844654, 7.2090216, -27.8820610, 27.9040298
22: -22.9081020, 9.3565235, -22.9169216, 9.3352795, -31.3324432, 31.3616714
23: -19.3494568, 4.2817111, -19.3553371, 4.2665505, -22.3250275, 22.3489838
24: -26.7305069, -1.6853833, -26.7446899, -1.7033167, -21.4584961, 21.4894562
25: -13.2764921, 9.5079794, -13.2894669, 9.4940634, -21.3186760, 21.3464317
26: -28.9153404, 8.7948151, -28.9248924, 8.7738342, -37.6591949, 37.6897278
27: -28.5649872, 0.3341494, -28.5787582, 0.3144665, -24.4630508, 24.4973984
28: -18.5113602, 6.3289614, -18.5253201, 6.3101840, -23.9441452, 23.9769897
29: -32.0678482, 5.0709562, -32.0736885, 5.0510063, -35.7663422, 35.7925568
30: -18.4635620, 8.4137993, -18.4790478, 8.4088593, -25.7405930, 25.7617912
31: -17.9909782, 8.5126905, -18.0035553, 8.4968281, -25.0752411, 25.1034203
32: -21.3984528, 4.2297959, -21.3924599, 4.2308030, -22.2850571, 22.2881203
33: -39.2937431, 1.1090040, -39.2838020, 1.1052947, -32.2215424, 32.2151947
34: -30.8066311, 2.1925111, -30.7994518, 2.1914086, -27.8132172, 27.8073349
35: -30.2974701, 2.4756155, -30.2951641, 2.4739594, -26.2378616, 26.2369041
36: -31.7448349, 0.2049997, -31.7484112, 0.2045076, -24.5528107, 24.5580330
37: -47.3297844, -6.5161057, -47.3321114, -6.5173197, -32.5940781, 32.5993919
38: -40.6429367, -2.1671257, -40.6468010, -2.1681314, -27.7597122, 27.7652626
39: -50.5442314, -5.9491606, -50.5335236, -5.9516015, -34.4423752, 34.4326248
40: -41.6889801, -3.3860483, -41.6814117, -3.3858571, -31.7639313, 31.7591095
41: -31.1470261, -4.2253790, -31.1421261, -4.2256436, -20.0313148, 20.0364761
42: -18.1105137, 2.5852227, -18.0959969, 2.5851908, -19.5215912, 19.5286331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8439322
time: 23.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8965084, upper bound: 17.8521334
time: 22.48 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -28.6248589, 5.7917161, -28.6380157, 5.8148327, -34.4396896, 34.4297333
1: -15.2249651, 11.1345692, -15.2295427, 11.1485882, -26.3735542, 26.3641129
2: -12.3341227, 10.9004326, -12.3378067, 10.9181223, -23.2522449, 23.2382393
3: -8.9920559, 15.7845488, -8.9972763, 15.8120594, -24.8041153, 24.7818260
4: -12.7407284, 13.2634449, -12.7423801, 13.2767382, -26.0174675, 26.0058250
5: -9.9328194, 18.0412712, -9.9360504, 18.0695992, -28.0024185, 27.9773216
6: -27.4512749, -2.9565563, -27.4666290, -2.9155025, -18.9824257, 19.0024490
7: -13.2522469, 17.7096786, -13.2591839, 17.7191658, -30.9714127, 30.9688625
8: -16.9918461, 15.8006840, -16.9972324, 15.8194408, -31.7859802, 31.7647209
9: -12.2539158, 13.5706501, -12.2579803, 13.5951052, -21.3952713, 21.3702011
10: -13.0724411, 24.7208862, -13.0831833, 24.7632484, -34.6690674, 34.6251602
11: -22.6842117, 12.8432598, -22.7294579, 12.8470221, -33.8614578, 33.9064484
12: -20.8568516, 15.4393024, -20.8668365, 15.4790897, -36.1931801, 36.1625290
13: -21.0790939, 11.3022146, -21.0844555, 11.3635483, -25.9146957, 25.8511543
14: -43.0309601, 3.4184942, -43.0897903, 3.4394655, -34.3532867, 34.4080467
15: -15.1042242, 9.8567410, -15.1185455, 9.8835354, -24.4599304, 24.4206505
16: -21.1354446, 13.1512470, -21.1482697, 13.1632500, -33.3405685, 33.3601685
17: -33.8478088, 27.4964104, -33.8979797, 27.5134811, -52.4518738, 52.4908524
18: -17.6537361, 7.9876804, -17.7061882, 7.9972820, -24.3986244, 24.4475460
19: -20.0828648, 2.0487192, -20.1325226, 2.0492084, -21.5163422, 21.5714073
20: -10.1486788, 10.3004560, -10.1857796, 10.3046150, -19.7173920, 19.7550888
21: -20.6764431, 7.2282705, -20.7247219, 7.2305646, -27.9070072, 27.9529915
22: -22.9095840, 9.3732662, -22.9635201, 9.3772745, -31.3699036, 31.4247437
23: -19.3514843, 4.2938457, -19.4027615, 4.2959404, -22.3518219, 22.4084206
24: -26.7312412, -1.6713181, -26.7988777, -1.6699791, -21.4816475, 21.5593224
25: -13.2778168, 9.5191116, -13.3217354, 9.5213547, -21.3428535, 21.3908119
26: -28.9169998, 8.8112259, -28.9842949, 8.8217239, -37.7041779, 37.7723312
27: -28.5672913, 0.3502049, -28.6444073, 0.3542566, -24.5009079, 24.5787544
28: -18.5131950, 6.3437672, -18.5866470, 6.3443203, -23.9736328, 24.0531540
29: -32.0695267, 5.0870047, -32.1261826, 5.0885592, -35.7995300, 35.8612976
30: -18.4668312, 8.4185257, -18.5230732, 8.4204922, -25.7534332, 25.8117104
31: -17.9934349, 8.5253897, -18.0454655, 8.5276699, -25.1025352, 25.1581573
32: -21.4090462, 4.2303343, -21.4202213, 4.2598634, -22.3014984, 22.3149338
33: -39.3105087, 1.1127791, -39.3292999, 1.1637740, -32.2992554, 32.2593346
34: -30.8177795, 2.1941690, -30.8350887, 2.2228141, -27.8571625, 27.8399124
35: -30.3034744, 2.4774184, -30.3194160, 2.4957423, -26.2692223, 26.2648506
36: -31.7487450, 0.2058554, -31.7600861, 0.2117116, -24.5636597, 24.5693970
37: -47.3414764, -6.5141015, -47.3665161, -6.4818010, -32.6427536, 32.6292801
38: -40.6494141, -2.1654816, -40.6639442, -2.1457610, -27.7892380, 27.7775860
39: -50.5586472, -5.9468193, -50.5738182, -5.9060626, -34.5034256, 34.4744263
40: -41.7028503, -3.3855834, -41.7177505, -3.3477540, -31.8163986, 31.7870178
41: -31.1583252, -4.2240992, -31.1718311, -4.1999035, -20.0337639, 20.0562363
42: -18.1253052, 2.5862384, -18.1324768, 2.6236815, -19.5243340, 19.5552559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8604029
time: 26.62 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8970870, upper bound: 17.8685409
time: 17.13 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -28.6701546, 5.8133774, -28.6226063, 5.8026686, -34.4728241, 34.4359818
1: -15.2474174, 11.1477461, -15.2241173, 11.1407557, -26.3881721, 26.3718643
2: -12.3705921, 10.9230404, -12.3333197, 10.9167252, -23.2873173, 23.2563591
3: -9.0263844, 15.8103991, -8.9774494, 15.8074055, -24.8337898, 24.7878494
4: -12.7649422, 13.2826672, -12.7413292, 13.2765408, -26.0414829, 26.0239964
5: -9.9673090, 18.0655270, -9.9210777, 18.0627747, -28.0300827, 27.9866047
6: -27.4580364, -2.9282441, -27.4299583, -2.9558005, -18.9887257, 19.0175629
7: -13.2679081, 17.7239208, -13.2468691, 17.7187729, -30.9866810, 30.9707909
8: -17.0260124, 15.8199205, -16.9919739, 15.8129330, -31.8163300, 31.7837143
9: -12.2884007, 13.5984821, -12.2394323, 13.5954399, -21.4302444, 21.3805656
10: -13.1217203, 24.7636452, -13.0467196, 24.7554531, -34.7026138, 34.6276855
11: -22.6993561, 12.8593121, -22.6868591, 12.8221779, -33.8558731, 33.8809700
12: -20.8564148, 15.4537525, -20.8377953, 15.4435701, -36.1527100, 36.1666946
13: -21.1152039, 11.3338385, -21.0538940, 11.3260593, -25.9117432, 25.8507805
14: -43.0959702, 3.4380703, -43.0327644, 3.4111180, -34.4051743, 34.3549576
15: -15.1488953, 9.8763571, -15.1057262, 9.8599005, -24.4928131, 24.4523849
16: -21.1567631, 13.1660471, -21.1190071, 13.1622772, -33.3666611, 33.3386154
17: -33.8980675, 27.5065269, -33.8561630, 27.4868984, -52.4831238, 52.4542542
18: -17.6785393, 8.0089531, -17.6736069, 7.9596877, -24.3846283, 24.4338818
19: -20.1030769, 2.0696311, -20.0964127, 2.0190277, -21.5106850, 21.5567970
20: -10.1697884, 10.3315687, -10.1640863, 10.2838736, -19.7160301, 19.7623558
21: -20.7033100, 7.2646275, -20.6950054, 7.2098513, -27.9131622, 27.9596329
22: -22.9308205, 9.3848877, -22.9240570, 9.3353157, -31.3519058, 31.3970871
23: -19.3662853, 4.3062649, -19.3608360, 4.2671871, -22.3383408, 22.3886414
24: -26.7577324, -1.6390696, -26.7544479, -1.7030888, -21.4817657, 21.5456123
25: -13.3042698, 9.5498962, -13.2994690, 9.4939337, -21.3432007, 21.4013557
26: -28.9394760, 8.8233624, -28.9326305, 8.7741241, -37.6821671, 37.7269745
27: -28.5961494, 0.3807120, -28.5899639, 0.3151135, -24.4899597, 24.5562057
28: -18.5408096, 6.3714557, -18.5365791, 6.3106842, -23.9736557, 24.0313759
29: -32.0877609, 5.0938301, -32.0790367, 5.0518103, -35.7862015, 35.8221893
30: -18.4989910, 8.4641209, -18.4920025, 8.4098692, -25.7728500, 25.8257332
31: -18.0211220, 8.5545692, -18.0138245, 8.4973164, -25.1031189, 25.1554413
32: -21.4138680, 4.2495317, -21.3962593, 4.2321596, -22.3000107, 22.3204155
33: -39.3202744, 1.1464715, -39.2922821, 1.1056280, -32.2441330, 32.2632065
34: -30.8222198, 2.2124052, -30.8031425, 2.1918373, -27.8281593, 27.8295784
35: -30.3144436, 2.4965887, -30.3000946, 2.4741759, -26.2532387, 26.2626724
36: -31.7655945, 0.2296076, -31.7545338, 0.2048814, -24.5691338, 24.5886574
37: -47.3649864, -6.4719105, -47.3444443, -6.5165844, -32.6274300, 32.6570129
38: -40.6710854, -2.1421161, -40.6557236, -2.1675367, -27.7834435, 27.8013458
39: -50.5624466, -5.9299588, -50.5387497, -5.9514713, -34.4583054, 34.4583359
40: -41.7081184, -3.3585114, -41.6871758, -3.3855448, -31.7904854, 31.8068275
41: -31.1660366, -4.1972404, -31.1477890, -4.2246618, -20.0479965, 20.0699577
42: -18.1226635, 2.5980701, -18.0972881, 2.5859604, -19.5278912, 19.5701942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8723867
time: 26.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8723867
time: 35.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -28.6718960, 5.8200903, -28.6385574, 5.8255267, -34.4974213, 34.4586487
1: -15.2481842, 11.1509409, -15.2297859, 11.1540375, -26.4022217, 26.3807259
2: -12.3713007, 10.9253578, -12.3380861, 10.9274769, -23.2987785, 23.2634430
3: -9.0333681, 15.8116112, -8.9978409, 15.8224106, -24.8557777, 24.8094521
4: -12.7653198, 13.2828350, -12.7430458, 13.2834749, -26.0487938, 26.0258808
5: -9.9730377, 18.0664330, -9.9367142, 18.0792656, -28.0523033, 28.0031471
6: -27.4761276, -2.9277835, -27.4746971, -2.9147234, -19.0035114, 19.0462608
7: -13.2705059, 17.7248116, -13.2595806, 17.7241726, -30.9946785, 30.9843922
8: -17.0268097, 15.8232508, -16.9979839, 15.8278294, -31.8303375, 31.7869110
9: -12.2952843, 13.5989647, -12.2585039, 13.6058083, -21.4471512, 21.3944454
10: -13.1338882, 24.7656708, -13.0839920, 24.7794991, -34.7469177, 34.6646881
11: -22.7059174, 12.8697987, -22.7358303, 12.8483086, -33.8846359, 33.9405823
12: -20.8677902, 15.4548893, -20.8697281, 15.4819193, -36.2046509, 36.1949844
13: -21.1274014, 11.3360348, -21.0857925, 11.3750820, -25.9754295, 25.8820381
14: -43.0979004, 3.4560242, -43.0926170, 3.4534836, -34.4356918, 34.4386253
15: -15.1492138, 9.8880043, -15.1191006, 9.8949671, -24.5178223, 24.4519310
16: -21.1630516, 13.1666050, -21.1484280, 13.1681194, -33.3728333, 33.3732224
17: -33.8988113, 27.5204716, -33.9028168, 27.5225468, -52.5131531, 52.5145645
18: -17.6794128, 8.0226240, -17.7152176, 7.9980173, -24.4209023, 24.4926033
19: -20.1059570, 2.0825729, -20.1405907, 2.0493927, -21.5374985, 21.6137924
20: -10.1725311, 10.3394375, -10.1943417, 10.3049927, -19.7385101, 19.8029671
21: -20.7067223, 7.2733431, -20.7352638, 7.2314067, -27.9381294, 28.0086060
22: -22.9322910, 9.4016628, -22.9706345, 9.3772888, -31.3893890, 31.4601288
23: -19.3683224, 4.3183947, -19.4082527, 4.2965608, -22.3651581, 22.4481239
24: -26.7584782, -1.6249676, -26.8086491, -1.6697669, -21.5049133, 21.6154785
25: -13.3056011, 9.5610247, -13.3317356, 9.5212336, -21.3674011, 21.4457321
26: -28.9410973, 8.8397665, -28.9920425, 8.8219948, -37.7271423, 37.8096161
27: -28.5984077, 0.3967385, -28.6556416, 0.3548717, -24.5278625, 24.6375198
28: -18.5426407, 6.3862658, -18.5978775, 6.3448200, -24.0032196, 24.1074944
29: -32.0894623, 5.1098595, -32.1315422, 5.0893106, -35.8193588, 35.8909225
30: -18.5022850, 8.4688749, -18.5360527, 8.4215021, -25.7857056, 25.8756180
31: -18.0236053, 8.5672817, -18.0557518, 8.5281115, -25.1304398, 25.2101860
32: -21.4244652, 4.2500839, -21.4239922, 4.2612348, -22.3164444, 22.3472023
33: -39.3370209, 1.1502461, -39.3378563, 1.1640844, -32.3218689, 32.3073578
34: -30.8333626, 2.2141027, -30.8387814, 2.2232337, -27.8721008, 27.8621597
35: -30.3204212, 2.4983783, -30.3243637, 2.4959579, -26.2845993, 26.2906189
36: -31.7695599, 0.2304204, -31.7662048, 0.2120895, -24.5800095, 24.6000710
37: -47.3766670, -6.4698892, -47.3789024, -6.4810777, -32.6761322, 32.6869621
38: -40.6775551, -2.1404681, -40.6729546, -2.1451840, -27.8129616, 27.8136826
39: -50.5768471, -5.9275632, -50.5790787, -5.9059186, -34.5193558, 34.5001221
40: -41.7219925, -3.3580394, -41.7234879, -3.3474708, -31.8429337, 31.8347397
41: -31.1773567, -4.1959672, -31.1774616, -4.1989074, -20.0504303, 20.0897083
42: -18.1374207, 2.5990887, -18.1337662, 2.6244373, -19.5306244, 19.5968361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=105, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1747

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8888801
time: 25.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8723867
time: 120.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 148.44 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8439322
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8965084, upper bound: 17.8521334
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8604029
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8970870, upper bound: 17.8685409
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8723867
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8723867
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8888801
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 148.44
Output dim: 10, lower bound: -17.8521763, upper bound: 17.8723867

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -28.6208611, 5.7848301, -28.6220703, 5.7919869, -34.4128494, 34.4068985
1: -15.2230043, 11.1312027, -15.2238770, 11.1353016, -26.3583069, 26.3550797
2: -12.3322029, 10.8980198, -12.3330297, 10.9073782, -23.2395821, 23.2310486
3: -8.9838629, 15.7829838, -8.9768772, 15.7970581, -24.7809219, 24.7598610
4: -12.7392941, 13.2630568, -12.7406645, 13.2698069, -26.0091019, 26.0037212
5: -9.9258585, 18.0401821, -9.9204540, 18.0531235, -27.9789810, 27.9606361
6: -27.4327888, -2.9584723, -27.4218636, -2.9565678, -18.9672928, 18.9327908
7: -13.2484112, 17.7085705, -13.2464981, 17.7137794, -30.9621906, 30.9550686
8: -16.9894962, 15.7969704, -16.9911880, 15.8045559, -31.7704315, 31.7613792
9: -12.2453203, 13.5698929, -12.2389221, 13.5847349, -21.3333588, 21.3560104
10: -13.0583334, 24.7185516, -13.0458775, 24.7392483, -34.5904083, 34.5878754
11: -22.6772385, 12.8320560, -22.6804962, 12.8208838, -33.8322487, 33.8378220
12: -20.8449669, 15.4368105, -20.8349228, 15.4407063, -36.1697922, 36.1330605
13: -21.0653191, 11.2996464, -21.0525723, 11.3145313, -25.8096313, 25.8195114
14: -43.0266838, 3.4003935, -43.0299377, 3.3970623, -34.2715683, 34.3241844
15: -15.1026468, 9.8447628, -15.1051655, 9.8484573, -24.4277077, 24.4207878
16: -21.1274948, 13.1504669, -21.1188145, 13.1574230, -33.3147202, 33.3253250
17: -33.8453903, 27.4821014, -33.8513298, 27.4778309, -52.3856659, 52.4291153
18: -17.6525192, 7.9727240, -17.6645584, 7.9589567, -24.3619881, 24.3498402
19: -20.0797253, 2.0344920, -20.0883484, 2.0188091, -21.4892426, 21.4933739
20: -10.1457138, 10.2911291, -10.1555195, 10.2834835, -19.6947250, 19.6975288
21: -20.6726952, 7.2179508, -20.6844654, 7.2090216, -27.8817177, 27.9024162
22: -22.9076939, 9.3548365, -22.9169216, 9.3352795, -31.3320236, 31.3558655
23: -19.3491287, 4.2807021, -19.3553371, 4.2665505, -22.3247299, 22.3415108
24: -26.7300034, -1.6871662, -26.7446899, -1.7033167, -21.4579926, 21.4502182
25: -13.2762108, 9.5066109, -13.2894669, 9.4940634, -21.3184395, 21.3423920
26: -28.9150391, 8.7929096, -28.9248924, 8.7738342, -37.6588898, 37.6671753
27: -28.5647316, 0.3323131, -28.5787582, 0.3144665, -24.4627609, 24.4524193
28: -18.5111942, 6.3275394, -18.5253201, 6.3101840, -23.9439697, 23.9639397
29: -32.0672493, 5.0692387, -32.0736885, 5.0510063, -35.7657394, 35.7868881
30: -18.4632130, 8.4123363, -18.4790478, 8.4088593, -25.7402496, 25.7490501
31: -17.9904308, 8.5112686, -18.0035553, 8.4968281, -25.0747070, 25.0843697
32: -21.3980827, 4.2287211, -21.3924599, 4.2308030, -22.2847366, 22.2851143
33: -39.2933464, 1.1077528, -39.2838020, 1.1052947, -32.2211304, 32.1971283
34: -30.8063202, 2.1913853, -30.7994518, 2.1914086, -27.8128967, 27.7827530
35: -30.2971001, 2.4746027, -30.2951641, 2.4739594, -26.2374878, 26.2199860
36: -31.7444763, 0.2034090, -31.7484112, 0.2045076, -24.5525055, 24.5271072
37: -47.3291855, -6.5174975, -47.3321114, -6.5173197, -32.5934677, 32.5664368
38: -40.6426315, -2.1688313, -40.6468010, -2.1681314, -27.7593536, 27.7070198
39: -50.5437698, -5.9502110, -50.5335236, -5.9516015, -34.4418564, 34.4112930
40: -41.6886597, -3.3870363, -41.6814117, -3.3858571, -31.7635803, 31.7319145
41: -31.1465969, -4.2263598, -31.1421261, -4.2256436, -20.0309067, 19.9949989
42: -18.1102505, 2.5848651, -18.0959969, 2.5851908, -19.5269184, 19.5276890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8882926, upper bound: 17.8078365
time: 23.47 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8882926, upper bound: 17.8078365
time: 18.05 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -28.6225662, 5.7915597, -28.6380157, 5.8148327, -34.4374008, 34.4295769
1: -15.2237740, 11.1344347, -15.2295427, 11.1485882, -26.3723621, 26.3639774
2: -12.3329277, 10.9003363, -12.3378067, 10.9181223, -23.2510490, 23.2381439
3: -8.9908361, 15.7841930, -8.9972763, 15.8120594, -24.8028946, 24.7814693
4: -12.7396803, 13.2632256, -12.7423801, 13.2767382, -26.0164185, 26.0056057
5: -9.9315863, 18.0410881, -9.9360504, 18.0695992, -28.0011864, 27.9771385
6: -27.4508953, -2.9580178, -27.4666290, -2.9155025, -18.9820709, 18.9614792
7: -13.2510023, 17.7094383, -13.2591839, 17.7191658, -30.9701691, 30.9686222
8: -16.9903431, 15.8002939, -16.9972324, 15.8194408, -31.7844009, 31.7645683
9: -12.2521992, 13.5703907, -12.2579803, 13.5951052, -21.3502731, 21.3699017
10: -13.0705194, 24.7205849, -13.0831833, 24.7632484, -34.6346970, 34.6248474
11: -22.6838074, 12.8425322, -22.7294579, 12.8470221, -33.8610306, 33.8973885
12: -20.8563766, 15.4379625, -20.8668365, 15.4790897, -36.2217216, 36.1613617
13: -21.0774803, 11.3018713, -21.0844555, 11.3635483, -25.8733101, 25.8507538
14: -43.0286217, 3.4183969, -43.0897903, 3.4394655, -34.3020782, 34.4078407
15: -15.1029644, 9.8564072, -15.1185455, 9.8835354, -24.4527054, 24.4203262
16: -21.1337738, 13.1510229, -21.1482697, 13.1632500, -33.3208771, 33.3599167
17: -33.8461533, 27.4960938, -33.8979797, 27.5134811, -52.4157562, 52.4893570
18: -17.6534042, 7.9863911, -17.7061882, 7.9972820, -24.3982697, 24.4085541
19: -20.0825844, 2.0474381, -20.1325226, 2.0492084, -21.5160980, 21.5503464
20: -10.1484499, 10.2989969, -10.1857796, 10.3046150, -19.7171936, 19.7380943
21: -20.6760941, 7.2266531, -20.7247219, 7.2305646, -27.9066582, 27.9513741
22: -22.9091492, 9.3715954, -22.9635201, 9.3772745, -31.3694687, 31.4189072
23: -19.3511715, 4.2928047, -19.4027615, 4.2959404, -22.3515320, 22.4009628
24: -26.7307320, -1.6730866, -26.7988777, -1.6699791, -21.4811287, 21.5200882
25: -13.2775421, 9.5177393, -13.3217354, 9.5213547, -21.3426018, 21.3867416
26: -28.9166756, 8.8093376, -28.9842949, 8.8217239, -37.7038651, 37.7498245
27: -28.5669937, 0.3483515, -28.6444073, 0.3542566, -24.5006409, 24.5337524
28: -18.5130386, 6.3423829, -18.5866470, 6.3443203, -23.9734650, 24.0400734
29: -32.0689240, 5.0852423, -32.1261826, 5.0885592, -35.7989197, 35.8556519
30: -18.4664783, 8.4170742, -18.5230732, 8.4204922, -25.7530899, 25.7989502
31: -17.9929047, 8.5239620, -18.0454655, 8.5276699, -25.1020012, 25.1391125
32: -21.4086723, 4.2292585, -21.4202213, 4.2598634, -22.3011627, 22.3119354
33: -39.3100700, 1.1115255, -39.3292999, 1.1637740, -32.2988510, 32.2412491
34: -30.8174496, 2.1930432, -30.8350887, 2.2228141, -27.8568344, 27.8153419
35: -30.3030987, 2.4764199, -30.3194160, 2.4957423, -26.2688560, 26.2479248
36: -31.7484512, 0.2042346, -31.7600861, 0.2117116, -24.5633621, 24.5384712
37: -47.3408813, -6.5154805, -47.3665161, -6.4818010, -32.6421356, 32.5963211
38: -40.6491051, -2.1672001, -40.6639442, -2.1457610, -27.7888794, 27.7193546
39: -50.5581589, -5.9478574, -50.5738182, -5.9060626, -34.5028992, 34.4531097
40: -41.7025452, -3.3865352, -41.7177505, -3.3477540, -31.8160248, 31.7597923
41: -31.1579018, -4.2250843, -31.1718311, -4.1999035, -20.0333519, 20.0147686
42: -18.1250286, 2.5858846, -18.1324768, 2.6236815, -19.5296669, 19.5543232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8882926, upper bound: 17.8243438
time: 28.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8888804, upper bound: 17.8243438
time: 27.96 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.5192871, 5.8024435, -28.5796242, 5.8220897, -34.3413773, 34.3820686
1: -15.1726112, 11.1344814, -15.2003937, 11.1499128, -26.3225250, 26.3348751
2: -12.2879858, 10.9101439, -12.3059864, 10.9236813, -23.2116661, 23.2161293
3: -8.9389076, 15.7874336, -8.9617596, 15.8156796, -24.7545872, 24.7491932
4: -12.6890182, 13.2606211, -12.7138748, 13.2771854, -25.9662037, 25.9744949
5: -9.8829985, 18.0465660, -9.9023533, 18.0741386, -27.9571381, 27.9489193
6: -27.4504662, -3.0426836, -27.4677486, -2.9590750, -18.9354630, 18.9267807
7: -13.1886024, 17.7053947, -13.2277966, 17.7184525, -30.9070549, 30.9331913
8: -16.9091816, 15.8029385, -16.9533787, 15.8227816, -31.7031097, 31.7204208
9: -12.1577454, 13.5720310, -12.2062798, 13.5994196, -21.3028221, 21.3142929
10: -12.9944210, 24.7291012, -13.0311089, 24.7692165, -34.5968475, 34.5759048
11: -22.6824379, 12.7751970, -22.7285995, 12.8120785, -33.8247299, 33.8376160
12: -20.8488274, 15.4122734, -20.8610229, 15.4655704, -36.1409454, 36.1229706
13: -21.0114746, 11.3078880, -21.0415192, 11.3675537, -25.8488197, 25.8092995
14: -42.9234962, 3.4342537, -43.0260849, 3.4481115, -34.2592201, 34.3515472
15: -15.0667343, 9.8594074, -15.0875797, 9.8868217, -24.4234428, 24.3912163
16: -21.0574379, 13.1453476, -21.1074562, 13.1624174, -33.2617111, 33.3087387
17: -33.7846451, 27.4958172, -33.8581123, 27.5143681, -52.3831635, 52.4387741
18: -17.6549358, 7.9163218, -17.7086258, 7.9576521, -24.3546028, 24.3760891
19: -20.0837402, 1.9827518, -20.1347294, 2.0115597, -21.4787369, 21.5092964
20: -10.1513367, 10.2308960, -10.1893177, 10.2634697, -19.6750526, 19.6859245
21: -20.6786766, 7.1505609, -20.7277927, 7.1847715, -27.8634491, 27.8783531
22: -22.9054985, 9.2684593, -22.9630966, 9.3264809, -31.3120575, 31.3189392
23: -19.3479729, 4.2384329, -19.4029789, 4.2662826, -22.3148499, 22.3611870
24: -26.7312012, -1.7651291, -26.8018475, -1.7232184, -21.4244652, 21.4679985
25: -13.2824488, 9.4519415, -13.3256311, 9.4795313, -21.3036995, 21.3279572
26: -28.9139500, 8.6989689, -28.9855309, 8.7682381, -37.6464920, 37.6620636
27: -28.5706501, 0.2527542, -28.6492882, 0.3002729, -24.4479103, 24.4883804
28: -18.5225334, 6.2817812, -18.5935364, 6.3051119, -23.9444427, 23.9990654
29: -32.0573120, 4.9674110, -32.1214027, 5.0350723, -35.7333984, 35.7365341
30: -18.4751968, 8.3534231, -18.5291672, 8.3775864, -25.7144547, 25.7494049
31: -17.9930840, 8.4594488, -18.0471172, 8.4872389, -25.0593033, 25.0935478
32: -21.3982105, 4.1747689, -21.4159584, 4.2326298, -22.2620850, 22.2558365
33: -39.3068848, 1.0585032, -39.3277855, 1.1286831, -32.2573395, 32.2054176
34: -30.8108368, 2.1426148, -30.8313522, 2.1951971, -27.8280830, 27.7883377
35: -30.2961922, 2.4305420, -30.3160324, 2.4696679, -26.2348862, 26.2140503
36: -31.7418938, 0.1171613, -31.7577858, 0.1685064, -24.5115852, 24.4795227
37: -47.3339920, -6.5727005, -47.3650589, -6.5203443, -32.5907860, 32.5654984
38: -40.6434250, -2.2532921, -40.6632195, -2.1889305, -27.7334824, 27.6865273
39: -50.5403175, -5.9955635, -50.5665474, -5.9327102, -34.4562454, 34.4183502
40: -41.6926422, -3.4184756, -41.7140884, -3.3711066, -31.7887497, 31.7642860
41: -31.1514874, -4.2967443, -31.1693611, -4.2377415, -19.9871597, 19.9777145
42: -18.1191750, 2.5433917, -18.1278820, 2.6028786, -19.4898510, 19.5316448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=105, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1748

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8464830, upper bound: 17.8442495
time: 16.86 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8519397, upper bound: 17.8715188
time: 404.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 423.89 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 423.89
Output dim: 10, lower bound: -17.8882926, upper bound: 17.8078365
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 423.89
Output dim: 10, lower bound: -17.8882926, upper bound: 17.8078365
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 423.89
Output dim: 10, lower bound: -17.8882926, upper bound: 17.8243438
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 423.89
Output dim: 10, lower bound: -17.8888804, upper bound: 17.8243438
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 423.89
Output dim: 10, lower bound: -17.8464830, upper bound: 17.8442495
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 423.89
Output dim: 10, lower bound: -17.8519397, upper bound: 17.8715188

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -28.6208611, 5.7848301, -28.4694901, 5.7743154, -34.3951759, 34.2543182
1: -15.2230043, 11.1312027, -15.1483097, 11.1188335, -26.3418388, 26.2795124
2: -12.3322029, 10.8980198, -12.2497234, 10.8921852, -23.2243881, 23.1477432
3: -8.9838629, 15.7829838, -8.8824291, 15.7728901, -24.7567520, 24.6654129
4: -12.7392941, 13.2630568, -12.6643639, 13.2475901, -25.9868851, 25.9274216
5: -9.9258585, 18.0401821, -9.8304195, 18.0332317, -27.9590912, 27.8706017
6: -27.4327888, -2.9584723, -27.3961563, -3.0713630, -18.8538361, 18.9494591
7: -13.2484112, 17.7085705, -13.1646214, 17.6943359, -30.9427471, 30.8731918
8: -16.9894962, 15.7969704, -16.8735523, 15.7842293, -31.7506409, 31.6387787
9: -12.2453203, 13.5698929, -12.1014404, 13.5577908, -21.3487167, 21.2185173
10: -13.0583334, 24.7185516, -12.9064407, 24.7026520, -34.5871811, 34.4478416
11: -22.6772385, 12.8320560, -22.6570091, 12.7263260, -33.7365685, 33.8225403
12: -20.8449669, 15.4368105, -20.8159943, 15.3980675, -36.0741692, 36.1030350
13: -21.0653191, 11.2996464, -20.9366341, 11.2863445, -25.8232422, 25.6998711
14: -43.0266838, 3.4003935, -42.8555603, 3.3752294, -34.3002052, 34.1534767
15: -15.1026468, 9.8447628, -15.0227032, 9.8198643, -24.4061661, 24.3340759
16: -21.1274948, 13.1504669, -21.0132217, 13.1361408, -33.3086777, 33.2210693
17: -33.8453903, 27.4821014, -33.7372208, 27.4532127, -52.3912506, 52.3095703
18: -17.6525192, 7.9727240, -17.6400375, 7.8526897, -24.2519760, 24.3637295
19: -20.0797253, 2.0344920, -20.0661163, 1.9190164, -21.3902435, 21.4919395
20: -10.1457138, 10.2911291, -10.1343174, 10.1749487, -19.5823326, 19.6926041
21: -20.6726952, 7.2179508, -20.6563683, 7.0862551, -27.7589493, 27.8743191
22: -22.9076939, 9.3548365, -22.8900871, 9.2020893, -31.1980896, 31.3337860
23: -19.3491287, 4.2807021, -19.3349876, 4.1865797, -22.2425804, 22.3287773
24: -26.7300034, -1.6871662, -26.7174263, -1.8434672, -21.3171692, 21.4608917
25: -13.2762108, 9.5066109, -13.2662983, 9.3849859, -21.2057190, 21.3244743
26: -28.9150391, 8.7929096, -28.8977242, 8.6330662, -37.5178146, 37.6614151
27: -28.5647316, 0.3323131, -28.5509605, 0.1704993, -24.3190956, 24.4701500
28: -18.5111942, 6.3275394, -18.5051727, 6.2057047, -23.8395844, 23.9565735
29: -32.0672493, 5.0692387, -32.0414963, 4.9085722, -35.6208191, 35.7600708
30: -18.4632130, 8.4123363, -18.4519558, 8.2934141, -25.6201706, 25.7348900
31: -17.9904308, 8.5112686, -17.9730453, 8.3890142, -24.9665604, 25.0719032
32: -21.3980827, 4.2287211, -21.3662281, 4.1555119, -22.1996231, 22.2649994
33: -39.2933464, 1.1077528, -39.2536011, 1.0135665, -32.1291580, 32.1848717
34: -30.8063202, 2.1913853, -30.7768879, 2.1199203, -27.7464371, 27.7894745
35: -30.2971001, 2.4746027, -30.2709007, 2.4061456, -26.1694794, 26.2126541
36: -31.7444763, 0.2034090, -31.7207565, 0.0912471, -24.4395142, 24.5326576
37: -47.3291855, -6.5174975, -47.2894173, -6.6200905, -32.4866829, 32.5535507
38: -40.6426315, -2.1688313, -40.6126747, -2.2809386, -27.6430054, 27.7320175
39: -50.5437698, -5.9502110, -50.4969139, -6.0195508, -34.3727951, 34.3956184
40: -41.6886597, -3.3870363, -41.6520996, -3.4462814, -31.7029610, 31.7277527
41: -31.1465969, -4.2263598, -31.1162891, -4.3264222, -19.9257622, 20.0143509
42: -18.1102505, 2.5848651, -18.0777454, 2.5294971, -19.4596806, 19.5123520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=104, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8300214, upper bound: 17.8021189
time: 23.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8874237, upper bound: 17.8070004
time: 27.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -28.6208611, 5.7848301, -28.6197968, 5.7918115, -34.4126740, 34.4046249
1: -15.2230043, 11.1312027, -15.2226610, 11.1351480, -26.3581524, 26.3538628
2: -12.3322029, 10.8980198, -12.3318377, 10.9072819, -23.2394848, 23.2298584
3: -8.9838629, 15.7829838, -8.9756699, 15.7966805, -24.7805443, 24.7586536
4: -12.7392941, 13.2630568, -12.7396231, 13.2695560, -26.0088501, 26.0026798
5: -9.9258585, 18.0401821, -9.9192190, 18.0529289, -27.9787865, 27.9594002
6: -27.4327888, -2.9584723, -27.4215088, -2.9580283, -18.9263172, 18.9324341
7: -13.2484112, 17.7085705, -13.2452583, 17.7135849, -30.9619961, 30.9538288
8: -16.9894962, 15.7969704, -16.9896832, 15.8041439, -31.7702942, 31.7597847
9: -12.2453203, 13.5698929, -12.2371912, 13.5844574, -21.3330650, 21.3110161
10: -13.0583334, 24.7185516, -13.0439720, 24.7389450, -34.5900879, 34.5535049
11: -22.6772385, 12.8320560, -22.6800709, 12.8201466, -33.8232155, 33.8373871
12: -20.8449669, 15.4368105, -20.8344345, 15.4393673, -36.1686096, 36.1615906
13: -21.0653191, 11.2996464, -21.0509109, 11.3141642, -25.8092422, 25.7781448
14: -43.0266838, 3.4003935, -43.0275803, 3.3969178, -34.2713699, 34.2729568
15: -15.1026468, 9.8447628, -15.1039124, 9.8481312, -24.4273834, 24.4135590
16: -21.1274948, 13.1504669, -21.1171684, 13.1571951, -33.3144684, 33.3056335
17: -33.8453903, 27.4821014, -33.8496284, 27.4775143, -52.3841858, 52.3929443
18: -17.6525192, 7.9727240, -17.6641941, 7.9576778, -24.3230209, 24.3495007
19: -20.0797253, 2.0344920, -20.0880699, 2.0175381, -21.4681854, 21.4931068
20: -10.1457138, 10.2911291, -10.1552973, 10.2820177, -19.6777382, 19.6973228
21: -20.6726952, 7.2179508, -20.6841125, 7.2074428, -27.8801384, 27.9020634
22: -22.9076939, 9.3548365, -22.9164925, 9.3336020, -31.3262100, 31.3554535
23: -19.3491287, 4.2807021, -19.3550186, 4.2655168, -22.3172302, 22.3412170
24: -26.7300034, -1.6871662, -26.7441769, -1.7050996, -21.4187698, 21.4496994
25: -13.2762108, 9.5066109, -13.2891932, 9.4926958, -21.3143845, 21.3421364
26: -28.9150391, 8.7929096, -28.9245872, 8.7719250, -37.6363831, 37.6668625
27: -28.5647316, 0.3323131, -28.5784817, 0.3126292, -24.4177361, 24.4521561
28: -18.5111942, 6.3275394, -18.5251827, 6.3087759, -23.9308853, 23.9637794
29: -32.0672493, 5.0692387, -32.0730896, 5.0492668, -35.7600937, 35.7862854
30: -18.4632130, 8.4123363, -18.4786987, 8.4074059, -25.7274857, 25.7486916
31: -17.9904308, 8.5112686, -18.0029964, 8.4954052, -25.0556412, 25.0838337
32: -21.3980827, 4.2287211, -21.3920937, 4.2297277, -22.2817192, 22.2847900
33: -39.2933464, 1.1077528, -39.2833786, 1.1040144, -32.2030640, 32.1967087
34: -30.8063202, 2.1913853, -30.7991142, 2.1903043, -27.7883301, 27.7824173
35: -30.2971001, 2.4746027, -30.2947845, 2.4729691, -26.2205658, 26.2195969
36: -31.7444763, 0.2034090, -31.7480812, 0.2029135, -24.5215759, 24.5268173
37: -47.3291855, -6.5174975, -47.3315620, -6.5186987, -32.5605087, 32.5658226
38: -40.6426315, -2.1688313, -40.6464424, -2.1698513, -27.7010956, 27.7066498
39: -50.5437698, -5.9502110, -50.5330467, -5.9525790, -34.4205551, 34.4107895
40: -41.6886597, -3.3870363, -41.6811295, -3.3868442, -31.7363472, 31.7315369
41: -31.1465969, -4.2263598, -31.1417332, -4.2266359, -19.9894466, 19.9945850
42: -18.1102505, 2.5848651, -18.0957222, 2.5848398, -19.5259857, 19.5330181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=104, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8300214, upper bound: 17.8021189
time: 24.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8874242, upper bound: 17.8070004
time: 25.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.6225662, 5.7915597, -28.4854202, 5.7971430, -34.4197083, 34.2769814
1: -15.2237740, 11.1344347, -15.1539650, 11.1321201, -26.3558941, 26.2883987
2: -12.3329277, 10.9003363, -12.2544861, 10.9028673, -23.2357941, 23.1548233
3: -8.9908361, 15.7841930, -8.9028254, 15.7878714, -24.7787075, 24.6870193
4: -12.7396803, 13.2632256, -12.6660614, 13.2544918, -25.9941711, 25.9292870
5: -9.9315863, 18.0410881, -9.8460503, 18.0497265, -27.9813118, 27.8871384
6: -27.4508953, -2.9580178, -27.4408913, -3.0302711, -18.8686180, 18.9781532
7: -13.2510023, 17.7094383, -13.1772776, 17.6997261, -30.9507294, 30.8867149
8: -16.9903431, 15.8002939, -16.8796024, 15.7991018, -31.7646408, 31.6419868
9: -12.2521992, 13.5703907, -12.1204939, 13.5681591, -21.3656349, 21.2323723
10: -13.0705194, 24.7205849, -12.9437218, 24.7266808, -34.6315384, 34.4848022
11: -22.6838074, 12.8425322, -22.7059708, 12.7524481, -33.7653427, 33.8821259
12: -20.8563766, 15.4379625, -20.8478699, 15.4364433, -36.1260986, 36.1312943
13: -21.0774803, 11.3018713, -20.9685421, 11.3353920, -25.8869514, 25.7311287
14: -43.0286217, 3.4183969, -42.9154510, 3.4176540, -34.3307037, 34.2371712
15: -15.1029644, 9.8564072, -15.0360708, 9.8549099, -24.4311867, 24.3336182
16: -21.1337738, 13.1510229, -21.0426102, 13.1419754, -33.3148117, 33.2556839
17: -33.8461533, 27.4960938, -33.7839050, 27.4888191, -52.4213562, 52.3698730
18: -17.6534042, 7.9863911, -17.6816940, 7.8910174, -24.2882462, 24.4224548
19: -20.0825844, 2.0474381, -20.1102791, 1.9493893, -21.4170914, 21.5489197
20: -10.1484499, 10.2989969, -10.1645851, 10.1960745, -19.6048050, 19.7331924
21: -20.6760941, 7.2266531, -20.6966248, 7.1078010, -27.7838955, 27.9232788
22: -22.9091492, 9.3715954, -22.9367027, 9.2440796, -31.2355499, 31.3968201
23: -19.3511715, 4.2928047, -19.3823891, 4.2159796, -22.2693901, 22.3882256
24: -26.7307320, -1.6730866, -26.7716141, -1.8101482, -21.3403244, 21.5307770
25: -13.2775421, 9.5177393, -13.2985601, 9.4122810, -21.2298660, 21.3688354
26: -28.9166756, 8.8093376, -28.9571342, 8.6809187, -37.5628433, 37.7440414
27: -28.5669937, 0.3483515, -28.6166534, 0.2102904, -24.3570023, 24.5514755
28: -18.5130386, 6.3423829, -18.5664902, 6.2398510, -23.8690948, 24.0327110
29: -32.0689240, 5.0852423, -32.0940437, 4.9460812, -35.6539612, 35.8288498
30: -18.4664783, 8.4170742, -18.4959984, 8.3050404, -25.6330032, 25.7847443
31: -17.9929047, 8.5239620, -18.0149670, 8.4198427, -24.9938507, 25.1266251
32: -21.4086723, 4.2292585, -21.3938866, 4.1845708, -22.2160873, 22.2918243
33: -39.3100700, 1.1115255, -39.2990990, 1.0721211, -32.2069168, 32.2289772
34: -30.8174496, 2.1930432, -30.8125114, 2.1513138, -27.7903557, 27.8219986
35: -30.3030987, 2.4764199, -30.2951317, 2.4279428, -26.2008781, 26.2405357
36: -31.7484512, 0.2042346, -31.7323704, 0.0984635, -24.4504013, 24.5440788
37: -47.3408813, -6.5154805, -47.3237991, -6.5845757, -32.5353775, 32.5834122
38: -40.6491051, -2.1672001, -40.6298027, -2.2585726, -27.6725235, 27.7443027
39: -50.5581589, -5.9478574, -50.5372543, -5.9739828, -34.4338760, 34.4374008
40: -41.7025452, -3.3865352, -41.6884346, -3.4081998, -31.7554321, 31.7556305
41: -31.1579018, -4.2250843, -31.1459484, -4.3006425, -19.9282265, 20.0341263
42: -18.1250286, 2.5858846, -18.1141720, 2.5679913, -19.4624481, 19.5390034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=104, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8607852, upper bound: 17.8186722
time: 17.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8880083, upper bound: 17.8235090
time: 26.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -28.6225662, 5.7915597, -28.6357269, 5.8146462, -34.4372139, 34.4272881
1: -15.2237740, 11.1344347, -15.2283411, 11.1484261, -26.3722000, 26.3627758
2: -12.3329277, 10.9003363, -12.3366184, 10.9180403, -23.2509689, 23.2369537
3: -8.9908361, 15.7841930, -8.9960699, 15.8117065, -24.8025436, 24.7802620
4: -12.7396803, 13.2632256, -12.7413158, 13.2764902, -26.0161705, 26.0045414
5: -9.9315863, 18.0410881, -9.9348412, 18.0693932, -28.0009804, 27.9759293
6: -27.4508953, -2.9580178, -27.4662743, -2.9169455, -18.9410877, 18.9611187
7: -13.2510023, 17.7094383, -13.2579517, 17.7189503, -30.9699516, 30.9673901
8: -16.9903431, 15.8002939, -16.9956951, 15.8190060, -31.7842941, 31.7629890
9: -12.2521992, 13.5703907, -12.2562370, 13.5948229, -21.3499794, 21.3248844
10: -13.0705194, 24.7205849, -13.0812950, 24.7629471, -34.6343994, 34.5904808
11: -22.6838074, 12.8425322, -22.7290535, 12.8463020, -33.8519821, 33.8969650
12: -20.8563766, 15.4379625, -20.8663559, 15.4777241, -36.2205505, 36.1898842
13: -21.0774803, 11.3018713, -21.0828495, 11.3631878, -25.8729095, 25.8093948
14: -43.0286217, 3.4183969, -43.0874557, 3.4392948, -34.3018723, 34.3566246
15: -15.1029644, 9.8564072, -15.1172943, 9.8832035, -24.4524040, 24.4131088
16: -21.1337738, 13.1510229, -21.1465797, 13.1630421, -33.3206100, 33.3402481
17: -33.8461533, 27.4960938, -33.8962936, 27.5131493, -52.4142456, 52.4531784
18: -17.6534042, 7.9863911, -17.7058105, 7.9959989, -24.3592987, 24.4082165
19: -20.0825844, 2.0474381, -20.1322651, 2.0479350, -21.4950142, 21.5501022
20: -10.1484499, 10.2989969, -10.1855564, 10.3031445, -19.7002106, 19.7378998
21: -20.6760941, 7.2266531, -20.7243786, 7.2289877, -27.9050827, 27.9510307
22: -22.9091492, 9.3715954, -22.9630775, 9.3756208, -31.3636780, 31.4184799
23: -19.3511715, 4.2928047, -19.4024544, 4.2948942, -22.3440628, 22.4006577
24: -26.7307320, -1.6730866, -26.7983742, -1.6717691, -21.4419136, 21.5195808
25: -13.2775421, 9.5177393, -13.3214512, 9.5199966, -21.3385391, 21.3864861
26: -28.9166756, 8.8093376, -28.9839706, 8.8197718, -37.6813812, 37.7495270
27: -28.5669937, 0.3483515, -28.6441097, 0.3524003, -24.4556580, 24.5334816
28: -18.5130386, 6.3423829, -18.5864754, 6.3429079, -23.9604187, 24.0399132
29: -32.0689240, 5.0852423, -32.1256027, 5.0868149, -35.7932510, 35.8550186
30: -18.4664783, 8.4170742, -18.5227451, 8.4190273, -25.7403221, 25.7985916
31: -17.9929047, 8.5239620, -18.0449429, 8.5262423, -25.0829620, 25.1385784
32: -21.4086723, 4.2292585, -21.4198494, 4.2588091, -22.2981606, 22.3116074
33: -39.3100700, 1.1115255, -39.3288956, 1.1625204, -32.2807999, 32.2408447
34: -30.8174496, 2.1930432, -30.8347645, 2.2216969, -27.8322601, 27.8149948
35: -30.3030987, 2.4764199, -30.3190784, 2.4947596, -26.2519608, 26.2475128
36: -31.7484512, 0.2042346, -31.7597733, 0.2101116, -24.5324402, 24.5381889
37: -47.3408813, -6.5154805, -47.3659630, -6.4831562, -32.6091652, 32.5957336
38: -40.6491051, -2.1672001, -40.6636543, -2.1474986, -27.7306366, 27.7189751
39: -50.5581589, -5.9478574, -50.5733299, -5.9071007, -34.4816132, 34.4525871
40: -41.7025452, -3.3865352, -41.7174644, -3.3487377, -31.7888298, 31.7594414
41: -31.1579018, -4.2250843, -31.1714268, -4.2008996, -19.9918938, 20.0143585
42: -18.1250286, 2.5858846, -18.1322041, 2.6233163, -19.5287285, 19.5596466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=104, inp2_unstable=104, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1748

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8306338, upper bound: 17.8295587
time: 22.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8519394, upper bound: 17.8341381
time: 19.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 43.85 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8300214, upper bound: 17.8021189
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8874237, upper bound: 17.8070004
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8300214, upper bound: 17.8021189
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8874242, upper bound: 17.8070004
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8607852, upper bound: 17.8186722
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8880083, upper bound: 17.8235090
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8306338, upper bound: 17.8295587
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 43.85
Output dim: 10, lower bound: -17.8519394, upper bound: 17.8341381

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.6174545, 5.7844815, -28.4682655, 5.7742538, -34.3917084, 34.2527466
1: -15.2212496, 11.1307545, -15.1476860, 11.1187019, -26.3399506, 26.2784405
2: -12.3308029, 10.8977194, -12.2492504, 10.8920832, -23.2228851, 23.1469688
3: -8.9822912, 15.7822695, -8.8819113, 15.7725830, -24.7548752, 24.6641808
4: -12.7380543, 13.2626305, -12.6639738, 13.2474327, -25.9854870, 25.9266052
5: -9.9243860, 18.0398788, -9.8299437, 18.0331516, -27.9575386, 27.8698235
6: -27.4319572, -2.9619255, -27.3958874, -3.0725532, -18.8521690, 18.9342289
7: -13.2470121, 17.7080517, -13.1641598, 17.6941833, -30.9411964, 30.8722115
8: -16.9877014, 15.7959213, -16.8729839, 15.7838869, -31.7476578, 31.6427536
9: -12.2430382, 13.5694370, -12.1006937, 13.5576153, -21.3177223, 21.2172604
10: -13.0557261, 24.7180252, -12.9056158, 24.7024612, -34.5662003, 34.4464264
11: -22.6765518, 12.8298874, -22.6567574, 12.7256250, -33.7351456, 33.8122902
12: -20.8437710, 15.4361753, -20.8155975, 15.3979206, -36.0877342, 36.0986862
13: -21.0627289, 11.2988930, -20.9358406, 11.2859993, -25.7861137, 25.6981354
14: -43.0231705, 3.3999405, -42.8543854, 3.3750963, -34.2691917, 34.1520233
15: -15.1011372, 9.8441257, -15.0222397, 9.8196297, -24.4041328, 24.3348351
16: -21.1250706, 13.1500120, -21.0124073, 13.1359968, -33.2950134, 33.2198029
17: -33.8432884, 27.4813423, -33.7365646, 27.4528656, -52.3857880, 52.3047791
18: -17.6513901, 7.9706631, -17.6396694, 7.8520317, -24.2500992, 24.3346634
19: -20.0791931, 2.0326161, -20.0659237, 1.9184177, -21.3891678, 21.4748039
20: -10.1453009, 10.2891445, -10.1341925, 10.1742678, -19.5812607, 19.6829147
21: -20.6721992, 7.2156992, -20.6561871, 7.0855441, -27.7577438, 27.8718872
22: -22.9069595, 9.3520670, -22.8898640, 9.2012005, -31.1965485, 31.3282166
23: -19.3486557, 4.2788734, -19.3347664, 4.1859870, -22.2415314, 22.3178711
24: -26.7291622, -1.6897302, -26.7170162, -1.8442988, -21.3155098, 21.4295273
25: -13.2757721, 9.5045624, -13.2661343, 9.3843231, -21.2046204, 21.3200912
26: -28.9144592, 8.7899208, -28.8975544, 8.6320477, -37.5161667, 37.6406555
27: -28.5642586, 0.3295703, -28.5507946, 0.1695890, -24.3177299, 24.4356117
28: -18.5109425, 6.3254261, -18.5050793, 6.2049856, -23.8385925, 23.9452515
29: -32.0662537, 5.0661221, -32.0411377, 4.9075890, -35.6188049, 35.7534409
30: -18.4626598, 8.4099331, -18.4517441, 8.2926407, -25.6188354, 25.7252579
31: -17.9892673, 8.5094166, -17.9725723, 8.3884163, -24.9647408, 25.0563354
32: -21.3973732, 4.2273121, -21.3659916, 4.1550546, -22.2031822, 22.2629395
33: -39.2922592, 1.1055169, -39.2532501, 1.0128722, -32.1262131, 32.1805038
34: -30.8057137, 2.1898017, -30.7767143, 2.1193933, -27.7428360, 27.7931023
35: -30.2963486, 2.4727859, -30.2706661, 2.4056134, -26.1670990, 26.2097206
36: -31.7438812, 0.2007439, -31.7205734, 0.0903761, -24.4384766, 24.5202942
37: -47.3279495, -6.5205011, -47.2889519, -6.6210866, -32.4844666, 32.5277100
38: -40.6419067, -2.1706872, -40.6124535, -2.2815628, -27.6401901, 27.7176933
39: -50.5420036, -5.9518580, -50.4964066, -6.0200462, -34.3689270, 34.3951416
40: -41.6875229, -3.3889318, -41.6517563, -3.4469347, -31.6992645, 31.7176628
41: -31.1458778, -4.2297344, -31.1159916, -4.3275046, -19.9246445, 19.9895134
42: -18.1098709, 2.5836186, -18.0776329, 2.5290813, -19.4634247, 19.5102577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=104, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8782487, upper bound: 17.8070004
time: 426.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8782487, upper bound: 17.8070004
time: 33.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.6174545, 5.7844815, -28.6186714, 5.7917223, -34.4091759, 34.4031525
1: -15.2212496, 11.1307545, -15.2220936, 11.1349907, -26.3562393, 26.3528481
2: -12.3308029, 10.8977194, -12.3313885, 10.9071922, -23.2379951, 23.2291069
3: -8.9822912, 15.7822695, -8.9751415, 15.7964725, -24.7787628, 24.7574120
4: -12.7380543, 13.2626305, -12.7392082, 13.2694159, -26.0074692, 26.0018387
5: -9.9243860, 18.0398788, -9.9187374, 18.0528221, -27.9772072, 27.9586163
6: -27.4319572, -2.9619255, -27.4212437, -2.9591632, -18.9248276, 18.9172401
7: -13.2470121, 17.7080517, -13.2447910, 17.7133884, -30.9603996, 30.9528427
8: -16.9877014, 15.7959213, -16.9890747, 15.8037758, -31.7673569, 31.7637405
9: -12.2430382, 13.5694370, -12.2364349, 13.5843124, -21.3020897, 21.3097420
10: -13.0557261, 24.7180252, -13.0431414, 24.7387543, -34.5690613, 34.5520630
11: -22.6765518, 12.8298874, -22.6798248, 12.8194475, -33.8217545, 33.8271561
12: -20.8437710, 15.4361753, -20.8340569, 15.4391365, -36.1821594, 36.1572266
13: -21.0627289, 11.2988930, -21.0500679, 11.3139057, -25.7721710, 25.7763367
14: -43.0231705, 3.3999405, -43.0264320, 3.3967433, -34.2400360, 34.2715263
15: -15.1011372, 9.8441257, -15.1034136, 9.8479004, -24.4253807, 24.4142761
16: -21.1250706, 13.1500120, -21.1163635, 13.1570435, -33.3010178, 33.3044014
17: -33.8432884, 27.4813423, -33.8489532, 27.4772301, -52.3791656, 52.3880081
18: -17.6513901, 7.9706631, -17.6638184, 7.9570041, -24.3211174, 24.3204498
19: -20.0791931, 2.0326161, -20.0878906, 2.0169339, -21.4670792, 21.4759750
20: -10.1453009, 10.2891445, -10.1551590, 10.2813654, -19.6766891, 19.6876411
21: -20.6721992, 7.2156992, -20.6839523, 7.2067089, -27.8789082, 27.8996506
22: -22.9069595, 9.3520670, -22.9162750, 9.3326979, -31.3246536, 31.3498611
23: -19.3486557, 4.2788734, -19.3548546, 4.2649231, -22.3161888, 22.3303375
24: -26.7291622, -1.6897302, -26.7438984, -1.7059431, -21.4170837, 21.4184456
25: -13.2757721, 9.5045624, -13.2890282, 9.4920197, -21.3132744, 21.3377609
26: -28.9144592, 8.7899208, -28.9243889, 8.7709446, -37.6347504, 37.6461639
27: -28.5642586, 0.3295703, -28.5783272, 0.3117313, -24.4164162, 24.4176559
28: -18.5109425, 6.3254261, -18.5250931, 6.3080959, -23.9299469, 23.9524612
29: -32.0662537, 5.0661221, -32.0727310, 5.0482769, -35.7580490, 35.7796707
30: -18.4626598, 8.4099331, -18.4785233, 8.4066353, -25.7261124, 25.7391281
31: -17.9892673, 8.5094166, -18.0026169, 8.4947872, -25.0538330, 25.0683537
32: -21.3973732, 4.2273121, -21.3918686, 4.2292709, -22.2852936, 22.2827187
33: -39.2922592, 1.1055169, -39.2829857, 1.1032825, -32.2000961, 32.1923981
34: -30.8057137, 2.1898017, -30.7989655, 2.1897688, -27.7846603, 27.7859993
35: -30.2963486, 2.4727859, -30.2945461, 2.4723897, -26.2182388, 26.2164879
36: -31.7438812, 0.2007439, -31.7478943, 0.2020345, -24.5205383, 24.5144577
37: -47.3279495, -6.5205011, -47.3311310, -6.5196810, -32.5582657, 32.5406647
38: -40.6419067, -2.1706872, -40.6462097, -2.1704421, -27.6982498, 27.6927891
39: -50.5420036, -5.9518580, -50.5324707, -5.9531593, -34.4166412, 34.4104729
40: -41.6875229, -3.3889318, -41.6807480, -3.3874574, -31.7327118, 31.7220840
41: -31.1458778, -4.2297344, -31.1414738, -4.2277489, -19.9881248, 19.9697571
42: -18.1098709, 2.5836186, -18.0956059, 2.5844178, -19.5297623, 19.5308819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=104, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8864667, upper bound: 17.8177170
time: 26.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8864667, upper bound: 17.8177170
time: 23.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.6191616, 5.7912331, -28.4841995, 5.7970500, -34.4162102, 34.2754326
1: -15.2220144, 11.1339617, -15.1533432, 11.1319561, -26.3539696, 26.2873039
2: -12.3315277, 10.9000282, -12.2540188, 10.9027767, -23.2343044, 23.1540470
3: -8.9892731, 15.7834673, -8.9023123, 15.7875919, -24.7768650, 24.6857796
4: -12.7384644, 13.2627831, -12.6656837, 13.2543421, -25.9928055, 25.9284668
5: -9.9301367, 18.0407715, -9.8455744, 18.0496140, -27.9797516, 27.8863449
6: -27.4500771, -2.9614596, -27.4405899, -3.0314636, -18.8669300, 18.9629250
7: -13.2496271, 17.7089348, -13.1768379, 17.6995716, -30.9491997, 30.8857727
8: -16.9885387, 15.7992754, -16.8790073, 15.7987566, -31.7616348, 31.6459770
9: -12.2499027, 13.5699215, -12.1197472, 13.5679874, -21.3346405, 21.2311325
10: -13.0679054, 24.7200527, -12.9428568, 24.7265129, -34.6104965, 34.4833755
11: -22.6831131, 12.8403387, -22.7057228, 12.7517681, -33.7639275, 33.8718948
12: -20.8551636, 15.4373140, -20.8474770, 15.4362583, -36.1396713, 36.1269112
13: -21.0749226, 11.3010597, -20.9677353, 11.3350983, -25.8498001, 25.7293854
14: -43.0251007, 3.4178839, -42.9142990, 3.4174981, -34.2996750, 34.2357063
15: -15.1014605, 9.8557587, -15.0355892, 9.8546648, -24.4291344, 24.3343697
16: -21.1313629, 13.1505623, -21.0417976, 13.1418142, -33.3011551, 33.2544060
17: -33.8439865, 27.4952946, -33.7832222, 27.4885025, -52.4158325, 52.3650589
18: -17.6522598, 7.9843240, -17.6813011, 7.8903584, -24.2863655, 24.3933640
19: -20.0820770, 2.0455744, -20.1100998, 1.9487826, -21.4159966, 21.5317383
20: -10.1480408, 10.2970142, -10.1644697, 10.1953859, -19.6037331, 19.7234802
21: -20.6756191, 7.2244248, -20.6964417, 7.1070967, -27.7827148, 27.9208660
22: -22.9084663, 9.3688383, -22.9364643, 9.2431784, -31.2339859, 31.3913116
23: -19.3506927, 4.2909837, -19.3821869, 4.2153864, -22.2683411, 22.3773308
24: -26.7298927, -1.6756573, -26.7712021, -1.8109612, -21.3386536, 21.4993782
25: -13.2771225, 9.5157194, -13.2983913, 9.4116364, -21.2288094, 21.3644600
26: -28.9160919, 8.8063107, -28.9569283, 8.6799297, -37.5611725, 37.7232971
27: -28.5665302, 0.3455453, -28.6164703, 0.2093935, -24.3556671, 24.5169678
28: -18.5127888, 6.3402443, -18.5663891, 6.2391243, -23.8681335, 24.0213928
29: -32.0679398, 5.0821228, -32.0936890, 4.9451056, -35.6519547, 35.8222122
30: -18.4659271, 8.4146614, -18.4957657, 8.3042641, -25.6316681, 25.7751579
31: -17.9917488, 8.5221157, -18.0145111, 8.4192524, -24.9920425, 25.1110573
32: -21.4079762, 4.2278671, -21.3936729, 4.1841254, -22.2196503, 22.2897568
33: -39.3089676, 1.1093116, -39.2987976, 1.0713882, -32.2039490, 32.2246094
34: -30.8168678, 2.1914859, -30.8123283, 2.1507926, -27.7867584, 27.8255959
35: -30.3023605, 2.4746056, -30.2948914, 2.4274111, -26.1984978, 26.2376137
36: -31.7478313, 0.2015781, -31.7321568, 0.0975842, -24.4493484, 24.5317116
37: -47.3396378, -6.5184855, -47.3233643, -6.5855446, -32.5331650, 32.5575829
38: -40.6483917, -2.1690164, -40.6295662, -2.2591739, -27.6697311, 27.7299881
39: -50.5564003, -5.9494853, -50.5366974, -5.9744921, -34.4299850, 34.4369049
40: -41.7014084, -3.3884368, -41.6880722, -3.4088616, -31.7517624, 31.7455673
41: -31.1571808, -4.2284584, -31.1456680, -4.3017502, -19.9271049, 20.0092964
42: -18.1246414, 2.5846343, -18.1140442, 2.5675697, -19.4661846, 19.5368919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=104, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=209, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1317

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8319336, upper bound: 17.8070004
time: 30.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8319336, upper bound: 17.8070004
time: 30.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 62.48 seconds
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 62.48
Output dim: 10, lower bound: -17.8782487, upper bound: 17.8070004
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 62.48
Output dim: 10, lower bound: -17.8782487, upper bound: 17.8070004
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 62.48
Output dim: 10, lower bound: -17.8864667, upper bound: 17.8177170
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 62.48
Output dim: 10, lower bound: -17.8864667, upper bound: 17.8177170
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 62.48
Output dim: 10, lower bound: -17.8319336, upper bound: 17.8070004
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 62.48
Output dim: 10, lower bound: -17.8319336, upper bound: 17.8070004

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.6174545, 5.7844815, -28.6169262, 5.7753549, -34.3928108, 34.4014091
1: -15.2212496, 11.1307545, -15.2213097, 11.1264906, -26.3477402, 26.3520641
2: -12.3308029, 10.8977194, -12.3306932, 10.8933477, -23.2241516, 23.2284126
3: -8.9822912, 15.7822695, -8.9740047, 15.7805367, -24.7628288, 24.7562752
4: -12.7380543, 13.2626305, -12.7382355, 13.2588224, -25.9968758, 26.0008659
5: -9.9243860, 18.0398788, -9.9176025, 18.0385380, -27.9629250, 27.9574814
6: -27.4319572, -2.9619255, -27.4083958, -2.9604845, -18.9236526, 18.9025192
7: -13.2470121, 17.7080517, -13.2436199, 17.7057686, -30.9527817, 30.9516716
8: -16.9877014, 15.7959213, -16.9877472, 15.7908173, -31.7539444, 31.7626495
9: -12.2430382, 13.5694370, -12.2352276, 13.5684290, -21.2860374, 21.3085594
10: -13.0557261, 24.7180252, -13.0410089, 24.7151051, -34.5452118, 34.5498505
11: -22.6765518, 12.8298874, -22.6686478, 12.8170557, -33.8192062, 33.8160172
12: -20.8437710, 15.4361753, -20.8286762, 15.4349556, -36.1788330, 36.1471596
13: -21.0627289, 11.2988930, -21.0475903, 11.2964687, -25.7543640, 25.7742805
14: -43.0231705, 3.3999405, -43.0215836, 3.3761120, -34.2192535, 34.2663765
15: -15.1011372, 9.8441257, -15.1015615, 9.8297720, -24.4068527, 24.4124908
16: -21.1250706, 13.1500120, -21.1145973, 13.1494026, -33.2939606, 33.3025894
17: -33.8432884, 27.4813423, -33.8418503, 27.4630775, -52.3647156, 52.3808823
18: -17.6513901, 7.9706631, -17.6504326, 7.9552803, -24.3190155, 24.3072109
19: -20.0791931, 2.0326161, -20.0756950, 2.0163567, -21.4664154, 21.4638023
20: -10.1453009, 10.2891445, -10.1419878, 10.2804737, -19.6758423, 19.6743240
21: -20.6721992, 7.2156992, -20.6680031, 7.2052155, -27.8774147, 27.8837013
22: -22.9069595, 9.3520670, -22.9053669, 9.3316269, -31.3235245, 31.3395462
23: -19.3486557, 4.2788734, -19.3462639, 4.2637620, -22.3149033, 22.3187294
24: -26.7291622, -1.6897302, -26.7286606, -1.7068863, -21.4161072, 21.4034767
25: -13.2757721, 9.5045624, -13.2741098, 9.4911575, -21.3124275, 21.3221931
26: -28.9144592, 8.7899208, -28.9124947, 8.7698975, -37.6336975, 37.6342010
27: -28.5642586, 0.3295703, -28.5615368, 0.3105063, -24.4152527, 24.4008331
28: -18.5109425, 6.3254261, -18.5086231, 6.3069162, -23.9287872, 23.9359131
29: -32.0662537, 5.0661221, -32.0643768, 5.0467243, -35.7565002, 35.7711411
30: -18.4626598, 8.4099331, -18.4587555, 8.4048634, -25.7246056, 25.7190399
31: -17.9892673, 8.5094166, -17.9866829, 8.4934406, -25.0524330, 25.0524120
32: -21.3973732, 4.2273121, -21.3845978, 4.2272673, -22.2832718, 22.2738838
33: -39.2922592, 1.1055169, -39.2712708, 1.1018953, -32.1986542, 32.1805496
34: -30.8057137, 2.1898017, -30.7922745, 2.1885953, -27.7833862, 27.7792587
35: -30.2963486, 2.4727859, -30.2867889, 2.4716005, -26.2174149, 26.2087250
36: -31.7438812, 0.2007439, -31.7382603, 0.2013865, -24.5198402, 24.5050888
37: -47.3279495, -6.5205011, -47.3124313, -6.5211663, -32.5569382, 32.5211029
38: -40.6419067, -2.1706872, -40.6327896, -2.1716805, -27.6969757, 27.6787739
39: -50.5420036, -5.9518580, -50.5239830, -5.9538226, -34.4157791, 34.4026260
40: -41.6875229, -3.3889318, -41.6694489, -3.3883729, -31.7317734, 31.7080116
41: -31.1458778, -4.2297344, -31.1322784, -4.2292228, -19.9869938, 19.9604683
42: -18.1098709, 2.5836186, -18.0914307, 2.5829587, -19.5281219, 19.5193386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=210, inp2_unstable=210, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=16, inp2_unstable=16, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1317

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1755

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -17.8684198, upper bound: 17.8119199
time: 32.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -17.8856256, upper bound: 17.8168760
time: 18.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 52.93 seconds
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 52.93
Output dim: 10, lower bound: -17.8684198, upper bound: 17.8119199
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 52.93
Output dim: 10, lower bound: -17.8856256, upper bound: 17.8168760
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 52.93
Output dim: 10, lower bound: -17.8864667, upper bound: 17.8177170

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 30.68 + 1773.82 = 1804.50 seconds
