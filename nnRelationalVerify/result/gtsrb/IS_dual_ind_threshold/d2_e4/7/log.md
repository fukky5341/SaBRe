## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 39.2744870991


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873)
1: (-44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5147743, 48.5147743)
2: (-35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6746597, 43.6746635)
3: (-46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2342491, 50.2342491)
4: (-36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7427902, 54.7427902)
5: (-49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0979233, 55.0979233)
6: (-43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734)
7: (-67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9659805, 55.9659729)
8: (-43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607)
9: (-21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806)
10: (-53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7385712, 68.7385712)
11: (-69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9733772, 46.9733734)
12: (-32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9903183, 59.9903183)
13: (-35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704)
14: (-105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5284882, 89.5284729)
15: (-35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9814987, 56.9814987)
16: (-61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0701180, 55.0701103)
17: (-123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6229477, 82.6229477)
18: (-47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825)
19: (-40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6668015, 39.6667938)
20: (-31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844)
21: (-53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9202728, 49.9202728)
22: (-54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8721924, 55.8721924)
23: (-32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7988663, 38.7988663)
24: (-26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1127548, 44.1127586)
25: (-23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7622910, 41.7622833)
26: (-45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034)
27: (-45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015)
28: (-36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9219666, 48.9219627)
29: (-65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3150940, 50.3150940)
30: (-43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718)
31: (-41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588)
32: (-38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566)
33: (-19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7451019, 73.7450943)
34: (-28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6961746, 70.6961670)
35: (-18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6688843, 71.6688919)
36: (-27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8722839, 73.8722839)
37: (-14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0141983, 55.0141869)
38: (-33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6856689, 87.6856613)
39: (-19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5259857, 77.5259933)
40: (-22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9437256, 61.9437218)
41: (-26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076)
42: (-35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.34 + 97.81 = 100.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 37, lower bound: -39.3138009, upper bound: 39.3138009

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3107079, upper bound: 39.2836883
time: 64.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3107079, upper bound: 39.3107078
time: 63.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 128.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 128.80
Output dim: 37, lower bound: -39.3107079, upper bound: 39.2836883
IS_A2, status: Status.UNKNOWN, split count: 1, time: 128.80
Output dim: 37, lower bound: -39.3107079, upper bound: 39.3107078

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -63.4468346, 8.5304680, -63.5201836, 8.6015177, -72.0483551, 72.0506516
1: -44.6072044, 6.6037846, -44.6620293, 6.6748724, -48.3697510, 48.3469963
2: -35.3698120, 10.6655025, -35.4217796, 10.7178125, -43.5358658, 43.5361176
3: -46.0125504, 7.4273577, -46.0672913, 7.4825344, -50.0881882, 50.0879440
4: -36.2491913, 19.6279068, -36.2867355, 19.6640854, -54.6484756, 54.6367760
5: -49.2076340, 11.0938301, -49.2607079, 11.1357718, -54.9642601, 54.9758568
6: -43.4943008, 19.2155361, -43.5223007, 19.2117062, -62.7060089, 62.7378387
7: -66.8612976, -0.3969288, -66.9338303, -0.3263111, -55.7666397, 55.7685585
8: -42.8251724, 24.9471970, -42.9064522, 25.0369873, -67.8621597, 67.8536530
9: -21.6449947, 16.4727173, -21.6652069, 16.4933052, -38.1382980, 38.1379242
10: -53.3033867, 17.4561062, -53.3396416, 17.4889221, -68.6442642, 68.6429138
11: -69.6349945, -11.5733147, -69.7002563, -11.5662451, -46.8102341, 46.8765869
12: -32.5819473, 30.0566673, -32.6249580, 30.1010036, -59.8752594, 59.8758621
13: -35.8591309, 37.4158859, -35.8780060, 37.4443283, -73.3034592, 73.2938919
14: -105.5168610, -11.0504494, -105.6074524, -10.9800129, -89.2956085, 89.3146667
15: -35.3313637, 21.9023190, -35.3561668, 21.9213219, -56.9072266, 56.8974495
16: -61.0581131, 2.3058567, -61.0939102, 2.3277168, -54.9756355, 54.9865303
17: -123.1228104, -17.6582375, -123.1735611, -17.6169624, -82.5131073, 82.5342102
18: -47.0121040, 24.1414833, -47.0775108, 24.2021885, -71.2142944, 71.2189941
19: -40.2570267, 1.7368941, -40.2753220, 1.7535601, -39.6279678, 39.6406593
20: -31.6924686, 5.3852158, -31.7175274, 5.3988376, -37.0913048, 37.1027451
21: -53.2704697, 0.2035265, -53.3014717, 0.2164650, -49.8467445, 49.8667107
22: -54.0522118, 6.0882502, -54.0810204, 6.1269207, -55.7497749, 55.7445068
23: -32.8415680, 8.2794590, -32.8562851, 8.2893600, -38.7343254, 38.7674026
24: -26.0941753, 18.4710751, -26.1176815, 18.4999924, -44.0353889, 44.0328293
25: -23.4279270, 19.6879234, -23.4668560, 19.7465382, -41.6233673, 41.6050034
26: -44.9758377, 25.2373276, -45.0500336, 25.2756634, -70.2515030, 70.2873611
27: -45.7630501, 10.9623756, -45.7833481, 10.9753389, -56.7383881, 56.7457237
28: -36.2023697, 14.3875103, -36.2234344, 14.4066286, -48.8418274, 48.8693848
29: -65.3661652, -6.1483669, -65.4023285, -6.1278028, -50.2289047, 50.2467232
30: -43.8240891, 14.4285612, -43.8478622, 14.4566383, -58.2807274, 58.2764244
31: -41.6330261, 2.8692498, -41.6643486, 2.8930187, -44.5260468, 44.5335999
32: -38.7813797, 22.6104507, -38.8022919, 22.6168842, -61.3982620, 61.4127426
33: -19.6794014, 60.1222725, -19.7339287, 60.1918488, -73.5518341, 73.5433350
34: -28.2301674, 47.6161766, -28.2801304, 47.6676331, -70.5504990, 70.5488586
35: -18.5915680, 56.1451263, -18.6073208, 56.1739426, -71.6228943, 71.5950623
36: -27.2828484, 48.3996048, -27.3353653, 48.4260025, -73.7595825, 73.7874603
37: -14.7328243, 48.4548988, -14.8062916, 48.5047569, -54.8305855, 54.8595276
38: -33.2820168, 57.8432693, -33.3176422, 57.8613586, -87.6016083, 87.6222916
39: -19.6967812, 65.9295654, -19.7042599, 65.9662933, -77.4481049, 77.4315491
40: -22.8643703, 42.3748932, -22.9029427, 42.4004173, -61.8499947, 61.8605919
41: -26.0595818, 26.3692913, -26.0876198, 26.3618011, -52.4213829, 52.4569092
42: -35.6518517, 19.6287003, -35.6738892, 19.6299419, -55.2817917, 55.3025894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2816303, upper bound: 39.2814540
time: 58.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3084744, upper bound: 39.2814540
time: 58.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -63.6098175, 8.6108551, -63.6122932, 8.6118526, -72.2216721, 72.2231445
1: -44.7285881, 6.6882753, -44.7309952, 6.6893091, -48.4759750, 48.5115471
2: -35.4862289, 10.7280264, -35.4884224, 10.7293291, -43.6159286, 43.6699753
3: -46.1367340, 7.4965000, -46.1393890, 7.4982567, -50.1900482, 50.2285576
4: -36.3293343, 19.6733360, -36.3312378, 19.6743813, -54.7356796, 54.7518845
5: -49.3236313, 11.1460085, -49.3258286, 11.1474905, -55.0686569, 55.0928650
6: -43.5465965, 19.2296352, -43.5556068, 19.2316189, -62.7782135, 62.7852402
7: -67.0221252, -0.3097744, -67.0278473, -0.3079891, -55.8743744, 55.9566803
8: -43.0063400, 25.0526028, -43.0085373, 25.0547867, -68.0611267, 68.0611420
9: -21.6855412, 16.5028954, -21.6865349, 16.5045719, -38.1901131, 38.1894302
10: -53.3766212, 17.5043163, -53.3783264, 17.5062141, -68.6968460, 68.7329712
11: -69.7550125, -11.5486012, -69.7699051, -11.5468597, -46.9016724, 46.9580345
12: -32.6367340, 30.1496696, -32.6386108, 30.1513557, -59.9879799, 59.9843559
13: -35.8894424, 37.4604416, -35.8904343, 37.4622650, -73.3517075, 73.3508759
14: -105.7122879, -10.9609509, -105.7151947, -10.9577751, -89.3898163, 89.5184021
15: -35.3834381, 21.9337826, -35.3854523, 21.9374580, -56.9678345, 56.9892578
16: -61.1266556, 2.3434210, -61.1343269, 2.3449230, -55.0333061, 55.0603294
17: -123.2302017, -17.6016159, -123.2335815, -17.5994110, -82.5928421, 82.6089401
18: -47.0981293, 24.2738571, -47.1010971, 24.2757092, -71.3738403, 71.3749542
19: -40.2905960, 1.7592149, -40.2924500, 1.7597318, -39.6684647, 39.6597519
20: -31.7318783, 5.4132576, -31.7327805, 5.4145346, -37.1464119, 37.1460381
21: -53.3252945, 0.2271309, -53.3297958, 0.2281694, -49.9028931, 49.9141388
22: -54.1039772, 6.1702051, -54.1061630, 6.1768465, -55.8620071, 55.8267784
23: -32.8656769, 8.2979183, -32.8672714, 8.2984285, -38.8315506, 38.7902756
24: -26.1298199, 18.5332355, -26.1309662, 18.5346909, -44.1114883, 44.1081734
25: -23.4780674, 19.8117962, -23.4791622, 19.8180008, -41.7543869, 41.7033691
26: -45.0758743, 25.3228836, -45.0781441, 25.3245964, -70.4004669, 70.4010315
27: -45.8006363, 10.9862537, -45.8024559, 10.9869995, -56.7876358, 56.7887115
28: -36.2316971, 14.4221954, -36.2323990, 14.4232187, -48.9499779, 48.9130974
29: -65.4382172, -6.1174097, -65.4403534, -6.1162720, -50.2870331, 50.3104668
30: -43.8576622, 14.4816723, -43.8585091, 14.4834795, -58.3411407, 58.3401794
31: -41.6742897, 2.9032092, -41.6750641, 2.9040456, -44.5783348, 44.5782738
32: -38.8131523, 22.6300964, -38.8198624, 22.6309395, -61.4440918, 61.4499588
33: -19.7508621, 60.2761307, -19.7530861, 60.2809486, -73.7362976, 73.6452942
34: -28.2997055, 47.7307320, -28.3018684, 47.7317924, -70.6900024, 70.6497650
35: -18.6207905, 56.2177963, -18.6225014, 56.2198563, -71.6615143, 71.6431885
36: -27.3526764, 48.4585648, -27.3548927, 48.4592285, -73.8670349, 73.8343887
37: -14.8347435, 48.5706673, -14.8393822, 48.5725136, -55.0024071, 54.9172935
38: -33.3419266, 57.8814125, -33.3453102, 57.8826637, -87.6748581, 87.6675720
39: -19.7192326, 66.0127563, -19.7214413, 66.0185852, -77.5155792, 77.4724884
40: -22.9237900, 42.4277115, -22.9267159, 42.4290962, -61.9368820, 61.9253731
41: -26.1164665, 26.3728981, -26.1258640, 26.3735485, -52.4900131, 52.4987640
42: -35.6928864, 19.6448479, -35.6994820, 19.6464367, -55.3393250, 55.3443298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2816303, upper bound: 39.3084743
time: 68.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3084744, upper bound: 39.3084743
time: 61.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 132.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 132.46
Output dim: 37, lower bound: -39.2816303, upper bound: 39.2814540
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 132.46
Output dim: 37, lower bound: -39.3084744, upper bound: 39.2814540
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 132.46
Output dim: 37, lower bound: -39.2816303, upper bound: 39.3084743
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 132.46
Output dim: 37, lower bound: -39.3084744, upper bound: 39.3084743

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -63.3655586, 8.5212069, -63.3618774, 8.5448322, -71.9103928, 71.8830872
1: -44.5466499, 6.5952997, -44.5457001, 6.6219311, -48.2564201, 48.2224236
2: -35.3254738, 10.6550121, -35.3390656, 10.6718931, -43.4452324, 43.4437332
3: -45.9931221, 7.4145603, -46.0285187, 7.4446344, -50.0336952, 50.0421906
4: -36.1808624, 19.6136093, -36.1540108, 19.5993042, -54.5138397, 54.4879417
5: -49.1699142, 11.0798073, -49.1884918, 11.0881338, -54.8831177, 54.8929138
6: -43.4870720, 19.1381702, -43.4763489, 19.0641441, -62.5512161, 62.6145172
7: -66.8020477, -0.4069405, -66.8206177, -0.3795547, -55.6493340, 55.6430206
8: -42.7436752, 24.9311314, -42.7512665, 24.9723167, -67.7159882, 67.6823959
9: -21.6126938, 16.4639053, -21.5931835, 16.4455910, -38.0582848, 38.0570908
10: -53.2075043, 17.4402122, -53.1575813, 17.4035664, -68.4631577, 68.4449005
11: -69.5941772, -11.5799122, -69.6206284, -11.6072521, -46.7293930, 46.7929420
12: -32.5634689, 30.0349731, -32.5761414, 30.0585690, -59.8009605, 59.7906990
13: -35.8460083, 37.3348541, -35.8113174, 37.2907944, -73.1368027, 73.1461716
14: -105.3730774, -11.0599737, -105.3350220, -11.0594158, -89.0663528, 89.0252151
15: -35.2603416, 21.8913689, -35.2187538, 21.8571644, -56.7711029, 56.7483444
16: -60.9877586, 2.2902470, -60.9579430, 2.2549028, -54.8327408, 54.8334885
17: -122.9745026, -17.6697769, -122.8926315, -17.7076454, -82.2739563, 82.2355042
18: -46.9506454, 24.1311340, -46.9594345, 24.1504784, -71.1011200, 71.0905685
19: -40.2256584, 1.7309098, -40.2092514, 1.7249527, -39.5722961, 39.5749283
20: -31.6773129, 5.3794832, -31.6727219, 5.3804231, -37.0577354, 37.0522041
21: -53.2265396, 0.1952000, -53.2091064, 0.1744747, -49.7637024, 49.7663116
22: -53.9835243, 6.0814161, -53.9455185, 6.0881796, -55.6424103, 55.6020203
23: -32.8196869, 8.2727242, -32.8082314, 8.2659798, -38.6887894, 38.7098694
24: -26.0637493, 18.4672508, -26.0551510, 18.4783955, -43.9805489, 43.9662018
25: -23.4089947, 19.6823387, -23.4150085, 19.7251854, -41.5812492, 41.5439148
26: -44.9221458, 25.2287617, -44.9391251, 25.2322731, -70.1544189, 70.1678848
27: -45.7294083, 10.9555759, -45.7158813, 10.9450989, -56.6745071, 56.6714554
28: -36.1846886, 14.3797998, -36.1857681, 14.3858337, -48.7984085, 48.8152351
29: -65.2903595, -6.1535664, -65.2581177, -6.1712513, -50.1090775, 50.0953827
30: -43.7951393, 14.4199524, -43.7883224, 14.4235878, -58.2187271, 58.2082748
31: -41.6169510, 2.8620176, -41.6226807, 2.8595529, -44.4765053, 44.4846992
32: -38.7652435, 22.5326004, -38.7368851, 22.4700050, -61.2352486, 61.2694855
33: -19.6584206, 60.0187759, -19.6301575, 59.9992828, -73.3381500, 73.3367004
34: -28.2119026, 47.5384064, -28.2094879, 47.5205307, -70.3832550, 70.4002991
35: -18.5777245, 56.0477066, -18.5300751, 55.9926987, -71.4280701, 71.4215851
36: -27.2691460, 48.2735977, -27.2490158, 48.1933823, -73.5126343, 73.5747528
37: -14.7089138, 48.4078941, -14.7118340, 48.4175186, -54.7169037, 54.7162437
38: -33.2611275, 57.7047997, -33.2034760, 57.6049461, -87.3218994, 87.3703690
39: -19.6754131, 65.7946243, -19.5953979, 65.7201462, -77.1772308, 77.1875534
40: -22.8446350, 42.2975845, -22.8309212, 42.2552567, -61.6850014, 61.7119865
41: -26.0486164, 26.3101826, -26.0356636, 26.2486591, -52.2972755, 52.3458481
42: -35.6405716, 19.5805187, -35.6285744, 19.5370178, -55.1775894, 55.2090912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2755064, upper bound: 39.2603418
time: 55.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2755064, upper bound: 39.2753279
time: 67.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -63.4456978, 8.5302591, -63.5173035, 8.6010046, -72.0466995, 72.0475616
1: -44.6061974, 6.6035776, -44.6594620, 6.6743412, -48.3682365, 48.3225899
2: -35.3694305, 10.6652689, -35.4210587, 10.7171869, -43.5345421, 43.5077133
3: -46.0115166, 7.4270544, -46.0646896, 7.4817495, -50.0856857, 50.0764427
4: -36.2480316, 19.6275082, -36.2837753, 19.6630592, -54.6462021, 54.6254501
5: -49.2068138, 11.0935249, -49.2586327, 11.1350040, -54.9621849, 54.9474258
6: -43.4940567, 19.2140732, -43.5217133, 19.2079849, -62.7020416, 62.7357864
7: -66.8606644, -0.3971920, -66.9321899, -0.3270473, -55.7651596, 55.7268753
8: -42.8237610, 24.9468117, -42.9028397, 25.0360508, -67.8598099, 67.8496552
9: -21.6443939, 16.4724846, -21.6638184, 16.4927006, -38.1370926, 38.1363029
10: -53.3017426, 17.4556103, -53.3354683, 17.4877243, -68.6413651, 68.6228333
11: -69.6341705, -11.5734701, -69.6981430, -11.5666695, -46.8087807, 46.8418045
12: -32.5815735, 30.0553360, -32.6240158, 30.0979652, -59.8681450, 59.8734703
13: -35.8587952, 37.4142303, -35.8771286, 37.4401474, -73.2989426, 73.2913589
14: -105.5140381, -11.0506735, -105.6002960, -10.9806528, -89.2920227, 89.2730408
15: -35.3301468, 21.9020462, -35.3530655, 21.9206314, -56.9052505, 56.8849030
16: -61.0567551, 2.3054647, -61.0905228, 2.3267336, -54.9732552, 54.9424133
17: -123.1202774, -17.6586323, -123.1671753, -17.6178665, -82.5095749, 82.4375992
18: -47.0109024, 24.1412201, -47.0744324, 24.2015095, -71.2124100, 71.2156525
19: -40.2563591, 1.7367229, -40.2736206, 1.7531013, -39.6265869, 39.6184311
20: -31.6920872, 5.3850775, -31.7165489, 5.3984900, -37.0905762, 37.1016273
21: -53.2694931, 0.2033405, -53.2989693, 0.2160206, -49.8449593, 49.8405571
22: -54.0507927, 6.0880909, -54.0773926, 6.1265078, -55.7479172, 55.7103729
23: -32.8409081, 8.2793102, -32.8546143, 8.2889729, -38.7319412, 38.7616730
24: -26.0932007, 18.4709702, -26.1152172, 18.4997120, -44.0365295, 44.0299911
25: -23.4274826, 19.6877747, -23.4657078, 19.7461433, -41.6220589, 41.6034241
26: -44.9748268, 25.2371101, -45.0474243, 25.2751045, -70.2499313, 70.2845306
27: -45.7621384, 10.9621944, -45.7810440, 10.9748697, -56.7370071, 56.7432404
28: -36.2018166, 14.3873644, -36.2220230, 14.4062309, -48.8381195, 48.8682060
29: -65.3645096, -6.1484842, -65.3981171, -6.1280813, -50.2268143, 50.1978912
30: -43.8231659, 14.4283772, -43.8455276, 14.4561634, -58.2793274, 58.2739029
31: -41.6325226, 2.8690524, -41.6631203, 2.8925447, -44.5250664, 44.5321732
32: -38.7808838, 22.6089420, -38.8010406, 22.6130791, -61.3939629, 61.4099808
33: -19.6788940, 60.1204109, -19.7326889, 60.1870804, -73.5241852, 73.5401993
34: -28.2296906, 47.6147766, -28.2789574, 47.6640549, -70.5124817, 70.5462341
35: -18.5911369, 56.1434479, -18.6062794, 56.1697083, -71.5970764, 71.5922928
36: -27.2824364, 48.3973465, -27.3343468, 48.4202042, -73.7350159, 73.7841492
37: -14.7321987, 48.4540443, -14.8046989, 48.5025902, -54.7967186, 54.8569946
38: -33.2815628, 57.8406372, -33.3164368, 57.8546677, -87.5567169, 87.6184540
39: -19.6961937, 65.9270172, -19.7028465, 65.9597931, -77.4049072, 77.4275970
40: -22.8637810, 42.3735008, -22.9014664, 42.3968658, -61.8055344, 61.8579712
41: -26.0592823, 26.3681412, -26.0868797, 26.3588524, -52.4181366, 52.4550209
42: -35.6515961, 19.6277122, -35.6732864, 19.6274185, -55.2790146, 55.3009987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2603418
time: 49.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2753279
time: 64.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.5284996, 8.6016273, -63.4539528, 8.5551987, -72.0836945, 72.0555801
1: -44.6680374, 6.6797743, -44.6146698, 6.6363621, -48.3626328, 48.3869743
2: -35.4418983, 10.7175341, -35.4057236, 10.6833982, -43.5252686, 43.5776215
3: -46.1173286, 7.4836721, -46.1006088, 7.4603395, -50.1355743, 50.1827850
4: -36.2610168, 19.6590614, -36.1985397, 19.6095772, -54.6010666, 54.6031227
5: -49.2859116, 11.1319799, -49.2536278, 11.0998497, -54.9875069, 55.0099411
6: -43.5393448, 19.1522903, -43.5096092, 19.0840721, -62.6234169, 62.6618996
7: -66.9628754, -0.3198166, -66.9146194, -0.3612461, -55.7570992, 55.8311539
8: -42.9248924, 25.0365448, -42.8533478, 24.9900761, -67.9149704, 67.8898926
9: -21.6532383, 16.4940834, -21.6145096, 16.4568615, -38.1100998, 38.1085930
10: -53.2807503, 17.4884224, -53.1962929, 17.4208431, -68.5157471, 68.5349426
11: -69.7141876, -11.5552216, -69.6902618, -11.5878916, -46.8208084, 46.8743668
12: -32.6182327, 30.1279621, -32.5897789, 30.1089344, -59.9136276, 59.8991699
13: -35.8762741, 37.3794327, -35.8237534, 37.3087349, -73.1850128, 73.2031860
14: -105.5685272, -10.9704561, -105.4427872, -11.0371094, -89.1605148, 89.2289810
15: -35.3124352, 21.9228535, -35.2480011, 21.8732986, -56.8316956, 56.8401070
16: -61.0563202, 2.3278494, -60.9983292, 2.2721157, -54.8904152, 54.9073029
17: -123.0819016, -17.6130886, -122.9526367, -17.6900425, -82.3537292, 82.3102417
18: -47.0366745, 24.2634697, -46.9830246, 24.2240200, -71.2606964, 71.2464905
19: -40.2592278, 1.7532220, -40.2263680, 1.7311335, -39.6127968, 39.5940247
20: -31.7167301, 5.4075203, -31.6879654, 5.3961029, -37.1128311, 37.0954857
21: -53.2813568, 0.2188053, -53.2374306, 0.1861782, -49.8199043, 49.8137321
22: -54.0352783, 6.1633720, -53.9706459, 6.1381016, -55.7546654, 55.6842728
23: -32.8437881, 8.2911854, -32.8192062, 8.2750473, -38.7860489, 38.7327194
24: -26.0993977, 18.5294285, -26.0684509, 18.5130882, -44.0566750, 44.0415611
25: -23.4591408, 19.8062019, -23.4273129, 19.7966557, -41.7122726, 41.6422577
26: -45.0221825, 25.3143005, -44.9672165, 25.2811947, -70.3033752, 70.2815170
27: -45.7670059, 10.9794321, -45.7350006, 10.9567575, -56.7237625, 56.7144318
28: -36.2140160, 14.4144726, -36.1947212, 14.4024191, -48.9065475, 48.8589478
29: -65.3623657, -6.1226196, -65.2961197, -6.1597404, -50.1671753, 50.1591034
30: -43.8287354, 14.4730282, -43.7989349, 14.4504318, -58.2791672, 58.2719650
31: -41.6582298, 2.8959942, -41.6333847, 2.8705969, -44.5288277, 44.5293808
32: -38.7970276, 22.5522766, -38.7544327, 22.4840622, -61.2810898, 61.3067093
33: -19.7298908, 60.1726227, -19.6493092, 60.0884132, -73.5226440, 73.4386673
34: -28.2814407, 47.6529579, -28.2312279, 47.5846786, -70.5227814, 70.5011597
35: -18.6069679, 56.1203499, -18.5452347, 56.0385742, -71.4667053, 71.4698029
36: -27.3389778, 48.3325958, -27.2685165, 48.2265930, -73.6200714, 73.6216736
37: -14.8108616, 48.5236664, -14.7449341, 48.4852982, -54.8887444, 54.7740326
38: -33.3209801, 57.7429390, -33.2310829, 57.6262207, -87.3951263, 87.4155731
39: -19.6979027, 65.8778076, -19.6125984, 65.7724533, -77.2446976, 77.2285004
40: -22.9040852, 42.3504028, -22.8547173, 42.2839661, -61.7719040, 61.7767830
41: -26.1055031, 26.3137932, -26.0738678, 26.2603912, -52.3658943, 52.3876610
42: -35.6815758, 19.5966396, -35.6541710, 19.5535278, -55.2351036, 55.2508087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2755064, upper bound: 39.2873740
time: 83.02 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2755064, upper bound: 39.3023596
time: 57.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.6086655, 8.6106577, -63.6094017, 8.6113882, -72.2200546, 72.2200623
1: -44.7275696, 6.6880760, -44.7284088, 6.6887951, -48.4744492, 48.4871216
2: -35.4858627, 10.7277832, -35.4877090, 10.7287092, -43.6145935, 43.6415825
3: -46.1357117, 7.4961967, -46.1367989, 7.4974632, -50.1875610, 50.2170372
4: -36.3281708, 19.6729527, -36.3282852, 19.6733398, -54.7333984, 54.7405777
5: -49.3228302, 11.1457052, -49.3237686, 11.1467218, -55.0665627, 55.0644226
6: -43.5463448, 19.2281799, -43.5550079, 19.2278976, -62.7742424, 62.7831879
7: -67.0214767, -0.3100681, -67.0262070, -0.3087273, -55.8728790, 55.9149818
8: -43.0049286, 25.0522270, -43.0049210, 25.0538139, -68.0587463, 68.0571442
9: -21.6849251, 16.5026550, -21.6851368, 16.5039730, -38.1888962, 38.1877899
10: -53.3749809, 17.5038490, -53.3741722, 17.5049744, -68.6939392, 68.7128601
11: -69.7541809, -11.5487957, -69.7677917, -11.5472946, -46.9001999, 46.9232330
12: -32.6363411, 30.1483307, -32.6376534, 30.1483307, -59.9808426, 59.9819717
13: -35.8891068, 37.4587860, -35.8895645, 37.4580956, -73.3471985, 73.3483505
14: -105.7094421, -10.9611893, -105.7080383, -10.9583673, -89.3862381, 89.4767761
15: -35.3822098, 21.9335232, -35.3823471, 21.9367676, -56.9658775, 56.9766884
16: -61.1253204, 2.3430395, -61.1309242, 2.3439398, -55.0309372, 55.0162125
17: -123.2276535, -17.6019745, -123.2271423, -17.6002979, -82.5893173, 82.5123138
18: -47.0969086, 24.2735825, -47.0980453, 24.2750473, -71.3719559, 71.3716278
19: -40.2899399, 1.7590427, -40.2907486, 1.7592807, -39.6670914, 39.6375237
20: -31.7314949, 5.4131141, -31.7317944, 5.4141850, -37.1456795, 37.1449089
21: -53.3243065, 0.2269621, -53.3272934, 0.2277327, -49.9011040, 49.8879738
22: -54.1025543, 6.1700287, -54.1025162, 6.1764345, -55.8601723, 55.7926636
23: -32.8650055, 8.2977676, -32.8655891, 8.2980518, -38.8291779, 38.7845421
24: -26.1288528, 18.5331306, -26.1285114, 18.5344238, -44.1126442, 44.1053505
25: -23.4776058, 19.8116379, -23.4780178, 19.8176060, -41.7530708, 41.7017822
26: -45.0748367, 25.3226776, -45.0755310, 25.3240337, -70.3988724, 70.3982086
27: -45.7997246, 10.9860668, -45.8001556, 10.9865313, -56.7862549, 56.7862244
28: -36.2311249, 14.4220390, -36.2309608, 14.4228096, -48.9462624, 48.9119110
29: -65.4365540, -6.1175404, -65.4361496, -6.1165667, -50.2849350, 50.2616348
30: -43.8567352, 14.4814682, -43.8561897, 14.4830112, -58.3397446, 58.3376579
31: -41.6737976, 2.9030147, -41.6738358, 2.9035640, -44.5773621, 44.5768509
32: -38.8126717, 22.6285934, -38.8185921, 22.6271210, -61.4397926, 61.4471855
33: -19.7503700, 60.2742691, -19.7518005, 60.2761993, -73.7086563, 73.6421661
34: -28.2992363, 47.7293243, -28.3006973, 47.7281990, -70.6519775, 70.6470947
35: -18.6204090, 56.2161484, -18.6214523, 56.2156219, -71.6356964, 71.6404724
36: -27.3522701, 48.4562950, -27.3538399, 48.4534416, -73.8424072, 73.8310852
37: -14.8341160, 48.5698242, -14.8377914, 48.5703583, -54.9685326, 54.9147606
38: -33.3414688, 57.8787880, -33.3440781, 57.8759537, -87.6299973, 87.6637039
39: -19.7186775, 66.0102005, -19.7200470, 66.0120926, -77.4723663, 77.4685059
40: -22.9232063, 42.4262962, -22.9252396, 42.4255676, -61.8924255, 61.9227676
41: -26.1161766, 26.3717308, -26.1251278, 26.3705997, -52.4867783, 52.4968567
42: -35.6926422, 19.6438522, -35.6988792, 19.6438904, -55.3365326, 55.3427315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2873740
time: 41.85 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3023597, upper bound: 39.3023596
time: 45.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 89.97 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.2755064, upper bound: 39.2603418
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.2755064, upper bound: 39.2753279
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2603418
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2753279
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.2755064, upper bound: 39.2873740
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.2755064, upper bound: 39.3023596
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.3023597, upper bound: 39.2873740
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 89.97
Output dim: 37, lower bound: -39.3023597, upper bound: 39.3023596

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -63.3054161, 8.5014381, -63.3254280, 8.5329008, -71.8383179, 71.8268661
1: -44.5034752, 6.5818644, -44.5196114, 6.6137657, -48.1921387, 48.1689796
2: -35.3067474, 10.6346798, -35.3276901, 10.6596861, -43.3487129, 43.3458023
3: -45.9826508, 7.3888502, -46.0222702, 7.4292259, -49.9119339, 49.9132423
4: -36.1277199, 19.5873451, -36.1220360, 19.5834942, -54.4000626, 54.3876266
5: -49.1577873, 11.0490561, -49.1812782, 11.0697050, -54.6863365, 54.6875610
6: -43.4754143, 19.0343590, -43.4693336, 19.0024796, -62.4778938, 62.5036926
7: -66.7685699, -0.4296761, -66.8005676, -0.3933353, -55.5246582, 55.5215073
8: -42.6860352, 24.9036751, -42.7168236, 24.9557476, -67.6417847, 67.6204987
9: -21.5825558, 16.4504776, -21.5749531, 16.4375210, -38.0200768, 38.0254288
10: -53.1053848, 17.4121208, -53.0968895, 17.3866482, -68.3392563, 68.3513412
11: -69.5383072, -11.5901442, -69.5873795, -11.6134586, -46.5904388, 46.6744461
12: -32.5272408, 30.0188999, -32.5544395, 30.0488529, -59.6523857, 59.6500816
13: -35.8297424, 37.2255669, -35.8015137, 37.2259102, -73.0556488, 73.0270844
14: -105.1819916, -11.0703068, -105.2214584, -11.0656338, -88.8672104, 88.9028397
15: -35.1969948, 21.8720932, -35.1804123, 21.8455772, -56.6460266, 56.6406136
16: -60.9182472, 2.2617388, -60.9165573, 2.2377243, -54.7661896, 54.7852058
17: -122.8070145, -17.6968327, -122.7931824, -17.7240028, -82.2066040, 82.2287140
18: -46.8403244, 24.1183186, -46.8937988, 24.1427479, -70.9830704, 71.0121155
19: -40.1811523, 1.7164483, -40.1827660, 1.7161946, -39.6076813, 39.6237946
20: -31.6458645, 5.3741341, -31.6536808, 5.3771787, -37.0230446, 37.0278168
21: -53.1467018, 0.1862736, -53.1614609, 0.1690836, -49.7327614, 49.7651367
22: -53.8581810, 6.0717516, -53.8708496, 6.0823050, -55.6704102, 55.6783829
23: -32.7801208, 8.2642651, -32.7846794, 8.2608852, -38.6636086, 38.6968918
24: -25.9675770, 18.4620247, -25.9978218, 18.4752045, -43.8527489, 43.8749352
25: -23.3658295, 19.6764259, -23.3886147, 19.7216053, -41.5821304, 41.5603790
26: -44.8009033, 25.2181721, -44.8668594, 25.2258968, -69.9820557, 70.0461731
27: -45.6361732, 10.9475746, -45.6601868, 10.9402542, -56.5764275, 56.6077614
28: -36.1475449, 14.3712015, -36.1636124, 14.3806772, -48.7688446, 48.7974892
29: -65.1716461, -6.1634092, -65.1873779, -6.1772776, -50.0410919, 50.0740929
30: -43.7047424, 14.4099140, -43.7342987, 14.4174366, -58.1221771, 58.1442108
31: -41.5848122, 2.8500366, -41.6035004, 2.8522902, -44.4371033, 44.4535370
32: -38.7283859, 22.4466496, -38.7147369, 22.4177361, -61.1461220, 61.1613846
33: -19.6309071, 59.9170036, -19.6136513, 59.9389191, -73.2370605, 73.2046967
34: -28.1834564, 47.4819832, -28.1923332, 47.4868927, -70.2806091, 70.2858810
35: -18.5572319, 55.9588509, -18.5176525, 55.9399986, -71.3670197, 71.3317871
36: -27.2465763, 48.1524925, -27.2353706, 48.1215820, -73.4138794, 73.4355164
37: -14.6690178, 48.3651352, -14.6878204, 48.3921814, -54.7016220, 54.6980515
38: -33.2307091, 57.5627785, -33.1851768, 57.5208130, -87.2200699, 87.2207031
39: -19.6387043, 65.6677475, -19.5732994, 65.6450272, -77.0718994, 77.0432129
40: -22.8039627, 42.2196617, -22.8064632, 42.2088776, -61.5644188, 61.5768356
41: -26.0347157, 26.2345467, -26.0272999, 26.2025375, -52.2372513, 52.2618484
42: -35.6253510, 19.5099392, -35.6194916, 19.4949379, -55.1202888, 55.1294327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2556231
time: 64.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2556231
time: 64.49 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -63.3943291, 8.5706425, -63.3505859, 8.5424900, -71.9368210, 71.9212265
1: -44.5618973, 6.6883116, -44.5427818, 6.6201010, -48.2638512, 48.3105049
2: -35.3246994, 10.6789217, -35.3311234, 10.6697493, -43.4488678, 43.4557304
3: -46.0180550, 7.4386244, -46.0270386, 7.4422970, -50.0714760, 50.0532913
4: -36.1971130, 19.6996651, -36.1463242, 19.5963554, -54.5227127, 54.5698662
5: -49.1875153, 11.1063938, -49.1866112, 11.0855646, -54.9170914, 54.9074898
6: -43.6016273, 19.1507912, -43.4739532, 19.0568581, -62.6584854, 62.6247444
7: -66.8059311, -0.3530598, -66.8115082, -0.3815689, -55.6503983, 55.6826859
8: -42.7530060, 24.9988976, -42.7431641, 24.9694939, -67.7225037, 67.7420654
9: -21.6667557, 16.5364609, -21.5900211, 16.4438438, -38.1105995, 38.1264801
10: -53.2217445, 17.5835762, -53.1507301, 17.4002762, -68.4683228, 68.5835266
11: -69.6072388, -11.4764872, -69.6157684, -11.6085548, -46.7341156, 46.8959389
12: -32.5963593, 30.0549240, -32.5732193, 30.0551987, -59.8367958, 59.8112144
13: -36.0054092, 37.3452415, -35.8094482, 37.2820015, -73.2874146, 73.1546936
14: -105.3993378, -10.8668680, -105.3218460, -11.0610161, -89.0702820, 89.2139435
15: -35.2747116, 22.0068302, -35.2149582, 21.8548698, -56.7749557, 56.8587494
16: -61.0227928, 2.4005909, -60.9524498, 2.2523651, -54.8568878, 54.9385910
17: -122.9880371, -17.4718914, -122.8817368, -17.7103958, -82.2557068, 82.4248734
18: -46.9595413, 24.2757187, -46.9510689, 24.1486931, -71.1082306, 71.2267914
19: -40.2470932, 1.7774186, -40.2055817, 1.7232447, -39.5886459, 39.6195412
20: -31.7086639, 5.4231901, -31.6688309, 5.3795419, -37.0882072, 37.0920219
21: -53.2585869, 0.2843399, -53.2027893, 0.1729622, -49.7928772, 49.8476410
22: -54.0068169, 6.1737766, -53.9373894, 6.0866127, -55.6531067, 55.6874580
23: -32.8389816, 8.3351536, -32.8041649, 8.2647705, -38.7042770, 38.7681236
24: -26.0804977, 18.5516033, -26.0476475, 18.4776497, -43.9952087, 44.0475197
25: -23.4456978, 19.7417698, -23.4098492, 19.7241325, -41.6147728, 41.5984993
26: -44.9475327, 25.3674660, -44.9302025, 25.2305470, -70.1780777, 70.2976685
27: -45.7484207, 11.0493889, -45.7078972, 10.9438210, -56.6922417, 56.7572861
28: -36.2047691, 14.4212980, -36.1815109, 14.3847780, -48.8156586, 48.8549271
29: -65.3060837, -6.0508490, -65.2503891, -6.1727619, -50.1083336, 50.1921158
30: -43.8144112, 14.5231762, -43.7791557, 14.4221296, -58.2365417, 58.3023300
31: -41.6473351, 2.9228711, -41.6195908, 2.8579483, -44.5052834, 44.5424614
32: -38.8752518, 22.5460663, -38.7334061, 22.4660130, -61.3412628, 61.2794724
33: -19.8517761, 60.0207291, -19.6266403, 59.9925690, -73.5257568, 73.3290939
34: -28.3042984, 47.5526352, -28.2054558, 47.5162964, -70.4712067, 70.4012909
35: -18.7215347, 56.0515671, -18.5271606, 55.9864807, -71.5661697, 71.4164734
36: -27.4276848, 48.2713547, -27.2466431, 48.1852913, -73.6638947, 73.5630951
37: -14.8480225, 48.4089813, -14.7074966, 48.4145737, -54.8575897, 54.7007408
38: -33.4644814, 57.7182312, -33.2004089, 57.5947876, -87.5162506, 87.3695068
39: -19.8797855, 65.7884827, -19.5917740, 65.7113876, -77.3753128, 77.1658249
40: -22.9576263, 42.3046036, -22.8261242, 42.2519073, -61.7841721, 61.6998291
41: -26.1655121, 26.3230705, -26.0335026, 26.2414169, -52.4069290, 52.3565750
42: -35.7317619, 19.5953197, -35.6266403, 19.5326576, -55.2644196, 55.2219620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2706229
time: 171.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2706229
time: 58.42 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -63.3853531, 8.5105305, -63.4807243, 8.5891361, -71.9744873, 71.9912567
1: -44.5627823, 6.5902042, -44.6331520, 6.6662588, -48.3037567, 48.2690048
2: -35.3506851, 10.6450014, -35.4098053, 10.7050238, -43.4378510, 43.4101524
3: -46.0007362, 7.4012947, -46.0582123, 7.4663401, -49.9636383, 49.9474754
4: -36.1946945, 19.6012840, -36.2516518, 19.6473598, -54.5319366, 54.5246811
5: -49.1946564, 11.0628405, -49.2513123, 11.1166735, -54.7651825, 54.7424507
6: -43.4822922, 19.1100426, -43.5146408, 19.1461525, -62.6284447, 62.6246834
7: -66.8275909, -0.4198761, -66.9123077, -0.3406620, -55.6395187, 55.6054573
8: -42.7654686, 24.9194317, -42.8679047, 25.0196190, -67.7850876, 67.7873383
9: -21.6141872, 16.4589520, -21.6455708, 16.4845600, -38.0987473, 38.1045227
10: -53.1992760, 17.4274502, -53.2745018, 17.4708118, -68.5171814, 68.5289383
11: -69.5781555, -11.5837479, -69.6647644, -11.5728769, -46.6698608, 46.7231102
12: -32.5454063, 30.0389862, -32.6024170, 30.0880318, -59.7192841, 59.7322350
13: -35.8424454, 37.3036423, -35.8673134, 37.3743057, -73.2167511, 73.1709595
14: -105.3216858, -11.0611229, -105.4857254, -10.9869556, -89.0915985, 89.1496201
15: -35.2665100, 21.8827744, -35.3145027, 21.9090691, -56.7795792, 56.7768402
16: -60.9869576, 2.2769375, -61.0489082, 2.3095608, -54.9061661, 54.8936119
17: -122.9524841, -17.6858253, -123.0674438, -17.6342773, -82.4418259, 82.4303436
18: -46.9001999, 24.1283455, -47.0085144, 24.1937408, -71.0939407, 71.1368561
19: -40.2115669, 1.7222037, -40.2469444, 1.7443390, -39.6613808, 39.6671219
20: -31.6605911, 5.3796768, -31.6975327, 5.3952370, -37.0558281, 37.0772095
21: -53.1893349, 0.1943779, -53.2510910, 0.2106066, -49.8135452, 49.8388214
22: -53.9251366, 6.0782919, -54.0025482, 6.1205673, -55.7754250, 55.7865334
23: -32.8009529, 8.2708225, -32.8307076, 8.2838573, -38.7067642, 38.7487335
24: -25.9964275, 18.4657135, -26.0574360, 18.4965420, -43.9080772, 43.9382248
25: -23.3840866, 19.6817741, -23.4396629, 19.7425098, -41.6227455, 41.6197701
26: -44.8532257, 25.2265053, -44.9748993, 25.2687016, -70.0954056, 70.1623611
27: -45.6681595, 10.9541922, -45.7248154, 10.9700441, -56.6382027, 56.6790085
28: -36.1642761, 14.3787098, -36.1995010, 14.4010239, -48.8079529, 48.8503838
29: -65.2452850, -6.1583118, -65.3270493, -6.1340866, -50.1582527, 50.1762199
30: -43.7320175, 14.4182968, -43.7908630, 14.4499912, -58.1820068, 58.2091599
31: -41.6002045, 2.8570466, -41.6437988, 2.8852658, -44.4854698, 44.5008469
32: -38.7439651, 22.5224800, -38.7789459, 22.5604019, -61.3043671, 61.3014259
33: -19.6512451, 60.0181541, -19.7160625, 60.1263580, -73.4225998, 73.4076385
34: -28.2012253, 47.5581512, -28.2618332, 47.6302490, -70.4094543, 70.4313965
35: -18.5704689, 56.0541306, -18.5938206, 56.1166153, -71.5353699, 71.5018387
36: -27.2597580, 48.2752991, -27.3206825, 48.3476639, -73.6354370, 73.6440277
37: -14.6920376, 48.4110565, -14.7805614, 48.4770508, -54.7810516, 54.8384514
38: -33.2509689, 57.6974602, -33.2981148, 57.7696304, -87.4536362, 87.4672165
39: -19.6593170, 65.7989655, -19.6807251, 65.8837280, -77.2984314, 77.2819672
40: -22.8227844, 42.2953568, -22.8768578, 42.3503151, -61.6844444, 61.7218399
41: -26.0452709, 26.2920284, -26.0784702, 26.3124046, -52.3576736, 52.3704987
42: -35.6362686, 19.5569592, -35.6640854, 19.5851765, -55.2214432, 55.2210464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2556231
time: 56.94 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2556231
time: 65.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -63.4736710, 8.5795727, -63.5055466, 8.5986404, -72.0723114, 72.0851212
1: -44.6213951, 6.6964502, -44.6565399, 6.6725025, -48.3755493, 48.4104118
2: -35.3681488, 10.6891804, -35.4125977, 10.7150307, -43.5374832, 43.5200539
3: -46.0359535, 7.4510422, -46.0631638, 7.4793339, -50.1231232, 50.0873871
4: -36.2641792, 19.7133579, -36.2760925, 19.6600208, -54.6548004, 54.7070007
5: -49.2242050, 11.1201725, -49.2567139, 11.1323938, -54.9931374, 54.9619484
6: -43.6084976, 19.2265663, -43.5192261, 19.2006378, -62.8091354, 62.7457924
7: -66.8638458, -0.3437252, -66.9226074, -0.3291321, -55.7656479, 55.7670250
8: -42.8325272, 25.0141869, -42.8944550, 25.0331573, -67.8656845, 67.9086456
9: -21.6982765, 16.5448780, -21.6605492, 16.4908543, -38.1891327, 38.2054291
10: -53.3155708, 17.5987930, -53.3283997, 17.4842854, -68.6459427, 68.7610626
11: -69.6471100, -11.4701405, -69.6931992, -11.5680389, -46.8129425, 46.9446793
12: -32.6145248, 30.0749798, -32.6210098, 30.0944977, -59.9041367, 59.8932381
13: -36.0180435, 37.4234085, -35.8751602, 37.4304581, -73.4485016, 73.2985687
14: -105.5390778, -10.8576756, -105.5862274, -10.9823160, -89.2947006, 89.4608154
15: -35.3442993, 22.0172081, -35.3491592, 21.9182625, -56.9088974, 56.9948883
16: -61.0914192, 2.4157133, -61.0848160, 2.3240519, -54.9967117, 55.0470238
17: -123.1334534, -17.4610062, -123.1560135, -17.6208420, -82.4907684, 82.6265564
18: -47.0194397, 24.2857132, -47.0657845, 24.1996498, -71.2190857, 71.3515015
19: -40.2774925, 1.7831511, -40.2697601, 1.7513461, -39.6419716, 39.6627960
20: -31.7234821, 5.4287453, -31.7125683, 5.3976121, -37.1210938, 37.1413116
21: -53.3012314, 0.2924175, -53.2924576, 0.2145042, -49.8734818, 49.9214592
22: -54.0738068, 6.1803179, -54.0689545, 6.1248837, -55.7582130, 55.7953491
23: -32.8599396, 8.3416700, -32.8503304, 8.2877197, -38.7475433, 38.8199730
24: -26.1094742, 18.5552883, -26.1072636, 18.4989719, -44.0506783, 44.1108398
25: -23.4642200, 19.7471123, -23.4599304, 19.7450333, -41.6555214, 41.6578674
26: -44.9998703, 25.3757515, -45.0382614, 25.2733593, -70.2732315, 70.4140167
27: -45.7805290, 11.0559483, -45.7725029, 10.9735527, -56.7540817, 56.8284531
28: -36.2216415, 14.4287815, -36.2174911, 14.4051228, -48.8548813, 48.9077950
29: -65.3797455, -6.0457983, -65.3898849, -6.1296177, -50.2255363, 50.2940559
30: -43.8417816, 14.5315161, -43.8359070, 14.4546509, -58.2964325, 58.3674240
31: -41.6627121, 2.9298210, -41.6598930, 2.8908887, -44.5536003, 44.5897141
32: -38.8905296, 22.6220856, -38.7973862, 22.6089077, -61.4994354, 61.4194717
33: -19.8721237, 60.1218796, -19.7290230, 60.1800003, -73.7112732, 73.5320129
34: -28.3219795, 47.6288261, -28.2748184, 47.6596909, -70.6001740, 70.5468292
35: -18.7347069, 56.1468735, -18.6032581, 56.1631279, -71.7345123, 71.5864792
36: -27.4407845, 48.3941727, -27.3318424, 48.4114151, -73.8853760, 73.7714615
37: -14.8709641, 48.4548988, -14.8002052, 48.4994354, -54.9369392, 54.8411102
38: -33.4845581, 57.8528900, -33.3132477, 57.8436584, -87.7498322, 87.6162720
39: -19.9003468, 65.9197540, -19.6991043, 65.9501114, -77.6018524, 77.4045868
40: -22.9763222, 42.3803558, -22.8964405, 42.3934135, -61.9049187, 61.8452415
41: -26.1759586, 26.3807316, -26.0846977, 26.3514080, -52.5273666, 52.4654312
42: -35.7424316, 19.6424408, -35.6713181, 19.6230202, -55.3654518, 55.3137589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2633150, upper bound: 39.2706229
time: 59.31 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2706229
time: 60.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -63.4683495, 8.5818253, -63.4175224, 8.5432749, -72.0116272, 71.9993439
1: -44.6248741, 6.6663399, -44.5885735, 6.6281900, -48.2983551, 48.3335228
2: -35.4231567, 10.6971893, -35.3943291, 10.6711998, -43.4287453, 43.4796448
3: -46.1068573, 7.4579554, -46.0943604, 7.4449234, -50.0137749, 50.0538330
4: -36.2079010, 19.6327953, -36.1665649, 19.5937824, -54.4873199, 54.5027695
5: -49.2737770, 11.1012373, -49.2464180, 11.0814085, -54.7907028, 54.8045654
6: -43.5276413, 19.0484982, -43.5025787, 19.0224152, -62.5500565, 62.5510788
7: -66.9293976, -0.3425274, -66.8945999, -0.3750038, -55.6324234, 55.7096634
8: -42.8672333, 25.0090523, -42.8189163, 24.9735107, -67.8407440, 67.8279724
9: -21.6231022, 16.4806557, -21.5962811, 16.4487915, -38.0718918, 38.0769348
10: -53.1786499, 17.4603615, -53.1356125, 17.4039459, -68.3918533, 68.4414291
11: -69.6583252, -11.5654316, -69.6570129, -11.5940857, -46.6818581, 46.7558670
12: -32.5819931, 30.1118698, -32.5680504, 30.0992088, -59.7651215, 59.7585526
13: -35.8600121, 37.2701416, -35.8139458, 37.2438736, -73.1038818, 73.0840912
14: -105.3775024, -10.9808159, -105.3292465, -11.0433426, -88.9614716, 89.1065521
15: -35.2491074, 21.9035950, -35.2096710, 21.8617210, -56.7066650, 56.7323799
16: -60.9868164, 2.2993679, -60.9569626, 2.2549639, -54.8239136, 54.8590393
17: -122.9144974, -17.6401405, -122.8531799, -17.7064018, -82.2863083, 82.3034821
18: -46.9263535, 24.2506504, -46.9174461, 24.2162743, -71.1426239, 71.1680984
19: -40.2147141, 1.7387643, -40.1998901, 1.7223711, -39.6481476, 39.6429024
20: -31.6852913, 5.4021711, -31.6689320, 5.3928776, -37.0781708, 37.0711021
21: -53.2015610, 0.2098837, -53.1898079, 0.1807909, -49.7889595, 49.8125877
22: -53.9100075, 6.1536732, -53.8960114, 6.1322336, -55.7827034, 55.7606163
23: -32.8042488, 8.2827158, -32.7956657, 8.2699547, -38.7608643, 38.7197495
24: -26.0032597, 18.5242043, -26.0110970, 18.5099201, -43.9289017, 43.9503098
25: -23.4159985, 19.8002853, -23.4009228, 19.7930717, -41.7131348, 41.6586990
26: -44.9009819, 25.3037186, -44.8949738, 25.2748299, -70.1291885, 70.1599045
27: -45.6737862, 10.9714413, -45.6792908, 10.9519463, -56.6257324, 56.6507339
28: -36.1768799, 14.4058695, -36.1725807, 14.3972664, -48.8769989, 48.8411789
29: -65.2436218, -6.1324739, -65.2253723, -6.1657648, -50.0993195, 50.1379051
30: -43.7383652, 14.4629898, -43.7449455, 14.4442768, -58.1826401, 58.2079353
31: -41.6260757, 2.8840246, -41.6142082, 2.8633265, -44.4894028, 44.4982338
32: -38.7601700, 22.4663277, -38.7322922, 22.4317818, -61.1919518, 61.1986198
33: -19.7023964, 60.0708771, -19.6328144, 60.0280495, -73.4216003, 73.3067017
34: -28.2529907, 47.5965271, -28.2140427, 47.5509987, -70.4200821, 70.3867569
35: -18.5864677, 56.0315323, -18.5328770, 55.9858665, -71.4056091, 71.3800430
36: -27.3164158, 48.2114716, -27.2548714, 48.1548080, -73.5213318, 73.4824371
37: -14.7710075, 48.4809265, -14.7209320, 48.4599648, -54.8734665, 54.7558403
38: -33.2906075, 57.6009369, -33.2127876, 57.5420952, -87.2933884, 87.2659988
39: -19.6612263, 65.7509537, -19.5905056, 65.6973038, -77.1393433, 77.0841217
40: -22.8634377, 42.2724380, -22.8302727, 42.2375641, -61.6512756, 61.6416512
41: -26.0915985, 26.2381897, -26.0655308, 26.2142811, -52.3058777, 52.3037186
42: -35.6663437, 19.5260696, -35.6450958, 19.5114517, -55.1777954, 55.1711655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2826577
time: 70.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2826577
time: 63.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -63.5572433, 8.6510954, -63.4426880, 8.5528402, -72.1100845, 72.0937805
1: -44.6832657, 6.7727995, -44.6117592, 6.6345425, -48.3700294, 48.4750710
2: -35.4411240, 10.7414742, -35.3977585, 10.6812725, -43.5289078, 43.5895958
3: -46.1422577, 7.5077381, -46.0991287, 7.4579773, -50.1733398, 50.1939087
4: -36.2772560, 19.7451382, -36.1908302, 19.6066322, -54.6099510, 54.6850357
5: -49.3034935, 11.1585884, -49.2517433, 11.0972853, -55.0214882, 55.0245399
6: -43.6538773, 19.1648884, -43.5072289, 19.0767746, -62.7306519, 62.6721191
7: -66.9667206, -0.2658443, -66.9055328, -0.3632336, -55.7581711, 55.8708725
8: -42.9341393, 25.1043491, -42.8452530, 24.9872742, -67.9214172, 67.9496002
9: -21.7072697, 16.5666122, -21.6113415, 16.4551105, -38.1623802, 38.1779556
10: -53.2949905, 17.6317749, -53.1894417, 17.4175682, -68.5208969, 68.6735687
11: -69.7272568, -11.4517870, -69.6853790, -11.5891895, -46.8255386, 46.9773941
12: -32.6511230, 30.1479187, -32.5868454, 30.1055546, -59.9495163, 59.9197083
13: -36.0356293, 37.3898087, -35.8218536, 37.2999802, -73.3356094, 73.2116623
14: -105.5948105, -10.7774181, -105.4296188, -11.0387602, -89.1644897, 89.4176712
15: -35.3268852, 22.0383263, -35.2442245, 21.8709869, -56.8356781, 56.9505348
16: -61.0913544, 2.4382143, -60.9928513, 2.2695608, -54.9146194, 55.0124245
17: -123.0954590, -17.4152279, -122.9417419, -17.6927834, -82.3354492, 82.4997025
18: -47.0455093, 24.4080238, -46.9746933, 24.2222328, -71.2677460, 71.3827209
19: -40.2806549, 1.7997465, -40.2227020, 1.7294393, -39.6291237, 39.6386566
20: -31.7481098, 5.4512157, -31.6840782, 5.3952289, -37.1433372, 37.1352921
21: -53.3134384, 0.3079338, -53.2311096, 0.1846781, -49.8490829, 49.8950882
22: -54.0586624, 6.2557125, -53.9625130, 6.1365461, -55.7654572, 55.7697144
23: -32.8630981, 8.3536177, -32.8151474, 8.2738361, -38.8015060, 38.7909546
24: -26.1161461, 18.6137409, -26.0609474, 18.5123596, -44.0713463, 44.1228523
25: -23.4958744, 19.8656158, -23.4221535, 19.7955875, -41.7458038, 41.6968422
26: -45.0476265, 25.4529934, -44.9583054, 25.2794857, -70.3271103, 70.4113007
27: -45.7860107, 11.0732212, -45.7269974, 10.9554901, -56.7415009, 56.8002167
28: -36.2341309, 14.4559555, -36.1904449, 14.4013596, -48.9237595, 48.8986320
29: -65.3780746, -6.0199194, -65.2883835, -6.1612606, -50.1664619, 50.2558098
30: -43.8480377, 14.5762386, -43.7898102, 14.4489536, -58.2969894, 58.3660507
31: -41.6886406, 2.9568911, -41.6302948, 2.8689799, -44.5576210, 44.5871849
32: -38.9070740, 22.5657234, -38.7509308, 22.4800644, -61.3871384, 61.3166542
33: -19.9233170, 60.1745567, -19.6457691, 60.0816956, -73.7102890, 73.4310837
34: -28.3738327, 47.6671524, -28.2271881, 47.5804367, -70.6106949, 70.5021362
35: -18.7507477, 56.1242218, -18.5423737, 56.0323639, -71.6047440, 71.4647369
36: -27.4975510, 48.3303452, -27.2661591, 48.2185059, -73.7713318, 73.6099777
37: -14.9499855, 48.5247612, -14.7405968, 48.4823532, -55.0294266, 54.7585258
38: -33.5243835, 57.7563629, -33.2280273, 57.6160889, -87.5895004, 87.4147797
39: -19.9022636, 65.8716507, -19.6089993, 65.7636719, -77.4427338, 77.2067566
40: -23.0170441, 42.3573532, -22.8499203, 42.2805901, -61.8710480, 61.7646103
41: -26.2223911, 26.3266830, -26.0717220, 26.2531662, -52.4755554, 52.3984070
42: -35.7727318, 19.6114998, -35.6522293, 19.5491352, -55.3218689, 55.2637291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2976518
time: 71.09 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2976518
time: 71.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -63.5482979, 8.5909414, -63.5728455, 8.5995255, -72.1478271, 72.1637878
1: -44.6841774, 6.6746712, -44.7021179, 6.6806822, -48.4099655, 48.4335365
2: -35.4670944, 10.7075100, -35.4764557, 10.7165470, -43.5178909, 43.5440178
3: -46.1249237, 7.4703827, -46.1303101, 7.4820337, -50.0655098, 50.0880470
4: -36.2748642, 19.6467304, -36.2961731, 19.6576424, -54.6191864, 54.6397972
5: -49.3106422, 11.1150341, -49.3164291, 11.1283789, -54.8695679, 54.8594398
6: -43.5345688, 19.1241608, -43.5479279, 19.1660862, -62.7006531, 62.6720886
7: -66.9884033, -0.3327427, -67.0063324, -0.3223553, -55.7472458, 55.7935867
8: -42.9466515, 25.0248337, -42.9699974, 25.0374031, -67.9840546, 67.9948273
9: -21.6547318, 16.4891167, -21.6668873, 16.4958401, -38.1505737, 38.1560059
10: -53.2725601, 17.4756756, -53.3132172, 17.4880905, -68.5697632, 68.6190033
11: -69.6981964, -11.5590477, -69.7344055, -11.5535069, -46.7612915, 46.8045425
12: -32.6001740, 30.1319656, -32.6160278, 30.1383972, -59.8319855, 59.8407173
13: -35.8727493, 37.3482399, -35.8797379, 37.3922615, -73.2650146, 73.2279816
14: -105.5171661, -10.9716091, -105.5934906, -10.9646673, -89.1858368, 89.3533096
15: -35.3185959, 21.9142609, -35.3437881, 21.9252281, -56.8402023, 56.8686790
16: -61.0555115, 2.3145084, -61.0893021, 2.3267736, -54.9638939, 54.9674263
17: -123.0599365, -17.6291885, -123.1275024, -17.6166687, -82.5215759, 82.5050964
18: -46.9862137, 24.2606945, -47.0321503, 24.2672749, -71.2534866, 71.2928467
19: -40.2451553, 1.7445335, -40.2640610, 1.7505045, -39.7018814, 39.6862106
20: -31.6999779, 5.4077282, -31.7127781, 5.4109306, -37.1109085, 37.1205063
21: -53.2441559, 0.2179642, -53.2794151, 0.2222996, -49.8697128, 49.8862610
22: -53.9769325, 6.1602440, -54.0276756, 6.1705208, -55.8877335, 55.8687630
23: -32.8250732, 8.2892780, -32.8416824, 8.2929306, -38.8040161, 38.7715912
24: -26.0320892, 18.5279026, -26.0707283, 18.5312424, -43.9842110, 44.0135880
25: -23.4342365, 19.8056393, -23.4519806, 19.8139725, -41.7537575, 41.7180939
26: -44.9532928, 25.3120346, -45.0030136, 25.3176346, -70.2425003, 70.2760773
27: -45.7057648, 10.9780407, -45.7439117, 10.9817114, -56.6874771, 56.7219543
28: -36.1936111, 14.4133854, -36.2084465, 14.4176102, -48.9160957, 48.8940849
29: -65.3172989, -6.1273994, -65.3650742, -6.1225615, -50.2164841, 50.2400894
30: -43.7656403, 14.4714031, -43.8015060, 14.4768314, -58.2424698, 58.2729111
31: -41.6414795, 2.8910122, -41.6545029, 2.8963008, -44.5377808, 44.5455170
32: -38.7757607, 22.5421486, -38.7964897, 22.5744591, -61.3502197, 61.3386383
33: -19.7227135, 60.1720047, -19.7352200, 60.2154846, -73.6070862, 73.5096359
34: -28.2707615, 47.6727066, -28.2835579, 47.6943550, -70.5489731, 70.5322647
35: -18.5997219, 56.1268196, -18.6090164, 56.1625214, -71.5739594, 71.5500488
36: -27.3296013, 48.3342514, -27.3401814, 48.3808746, -73.7428818, 73.6909714
37: -14.7939911, 48.5268250, -14.8136559, 48.5448303, -54.9528961, 54.8962288
38: -33.3108559, 57.7356071, -33.3257599, 57.7908974, -87.5269318, 87.5125656
39: -19.6818428, 65.8821564, -19.6979313, 65.9360504, -77.3659058, 77.3229065
40: -22.8822613, 42.3481522, -22.9006405, 42.3790131, -61.7713013, 61.7866325
41: -26.1021481, 26.2956734, -26.1167030, 26.3241634, -52.4263115, 52.4123764
42: -35.6772804, 19.5730782, -35.6896896, 19.6016731, -55.2789536, 55.2627678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2826577
time: 56.25 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2826577
time: 74.89 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -63.6366272, 8.6600380, -63.5976601, 8.6089973, -72.2456207, 72.2576981
1: -44.7427597, 6.7809429, -44.7255096, 6.6869783, -48.4817543, 48.5749741
2: -35.4845695, 10.7517262, -35.4792328, 10.7265368, -43.6175232, 43.6539345
3: -46.1601562, 7.5201883, -46.1352539, 7.4950409, -50.2249985, 50.2280083
4: -36.3442993, 19.7588177, -36.3206139, 19.6703281, -54.7420197, 54.8221474
5: -49.3401985, 11.1723766, -49.3218307, 11.1440973, -55.0975418, 55.0789719
6: -43.6607513, 19.2406483, -43.5525513, 19.2205601, -62.8813095, 62.7931976
7: -67.0246353, -0.2565556, -67.0166016, -0.3108120, -55.8733902, 55.9551849
8: -43.0136833, 25.1196232, -42.9965363, 25.0509186, -68.0646057, 68.1161575
9: -21.7387886, 16.5750351, -21.6818638, 16.5021191, -38.2409058, 38.2568970
10: -53.3887939, 17.6470070, -53.3671150, 17.5015602, -68.6985321, 68.8510818
11: -69.7671051, -11.4454527, -69.7628403, -11.5486631, -46.9043655, 47.0261116
12: -32.6693039, 30.1679726, -32.6346359, 30.1448555, -60.0168304, 60.0017395
13: -36.0482750, 37.4679565, -35.8875923, 37.4483948, -73.4966736, 73.3555450
14: -105.7344894, -10.7682123, -105.6939926, -10.9600687, -89.3888855, 89.6645050
15: -35.3964653, 22.0486984, -35.3784409, 21.9344139, -56.9696350, 57.0867386
16: -61.1599808, 2.4532928, -61.1252060, 2.3412552, -55.0544243, 55.1208229
17: -123.2408905, -17.4043236, -123.2159882, -17.6032085, -82.5705719, 82.7013245
18: -47.1053925, 24.4180336, -47.0893784, 24.2731800, -71.3785706, 71.5074158
19: -40.3110504, 1.8054786, -40.2868958, 1.7575078, -39.6824760, 39.6819153
20: -31.7629128, 5.4567804, -31.7278214, 5.4132948, -37.1762085, 37.1846008
21: -53.3560524, 0.3160305, -53.3207779, 0.2262087, -49.9296532, 49.9689255
22: -54.1256638, 6.2622604, -54.0940781, 6.1748476, -55.8705444, 55.8775940
23: -32.8840561, 8.3601389, -32.8613129, 8.2967882, -38.8447800, 38.8428116
24: -26.1451321, 18.6174355, -26.1205578, 18.5336456, -44.1268044, 44.1861725
25: -23.5143871, 19.8709679, -23.4722404, 19.8165092, -41.7865639, 41.7562218
26: -45.0999603, 25.4612808, -45.0663872, 25.3222885, -70.4222488, 70.5276642
27: -45.8181229, 11.0797997, -45.7916183, 10.9852180, -56.8033409, 56.8714180
28: -36.2509956, 14.4634600, -36.2264175, 14.4217129, -48.9629898, 48.9514847
29: -65.4517517, -6.0148506, -65.4279022, -6.1180544, -50.2836723, 50.3577728
30: -43.8754120, 14.5845947, -43.8465347, 14.4815130, -58.3569260, 58.4311295
31: -41.7040062, 2.9638276, -41.6706161, 2.9019132, -44.6059189, 44.6344452
32: -38.9223633, 22.6417122, -38.8149338, 22.6229630, -61.5453262, 61.4566460
33: -19.9436340, 60.2757416, -19.7481613, 60.2691078, -73.8957977, 73.6339722
34: -28.3915329, 47.7433281, -28.2965660, 47.7238312, -70.7396317, 70.6476593
35: -18.7639275, 56.2194977, -18.6184464, 56.2090225, -71.7730713, 71.6346741
36: -27.5106354, 48.4531555, -27.3513603, 48.4446411, -73.9928360, 73.8183289
37: -14.9729309, 48.5706787, -14.8332958, 48.5672073, -55.1087570, 54.8988953
38: -33.5444489, 57.8910179, -33.3408890, 57.8649139, -87.8231049, 87.6616135
39: -19.9228153, 66.0029449, -19.7162971, 66.0024109, -77.6692886, 77.4454956
40: -23.0357056, 42.4331245, -22.9201889, 42.4221115, -61.9917755, 61.9100189
41: -26.2328682, 26.3843384, -26.1229134, 26.3631592, -52.5960274, 52.5072517
42: -35.7834320, 19.6585922, -35.6969147, 19.6395092, -55.4229431, 55.3555069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 679

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2976518
time: 64.76 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2976518
time: 59.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 126.01 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2556231
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2556231
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2706229
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2706229
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2556231
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2556231
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2633150, upper bound: 39.2706229
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2706229
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2826577
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2826577
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2976518
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2708072, upper bound: 39.2976518
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2826577
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2826577
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2364538, upper bound: 39.2976518
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 126.01
Output dim: 37, lower bound: -39.2976518, upper bound: 39.2976518

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -63.3795662, 8.5092039, -63.4709091, 8.5868464, -71.9664154, 71.9801102
1: -44.5611420, 6.5880241, -44.6303978, 6.6625652, -48.2981758, 48.2408905
2: -35.3460312, 10.6432304, -35.4026718, 10.7020588, -43.4302979, 43.3829384
3: -46.0000305, 7.3954182, -46.0570145, 7.4564295, -49.9373550, 49.9201546
4: -36.1862335, 19.5992699, -36.2385483, 19.6439896, -54.5231934, 54.4515877
5: -49.1907463, 11.0607519, -49.2445374, 11.1132078, -54.7579575, 54.7169533
6: -43.4790649, 19.1052036, -43.5092506, 19.1377945, -62.6168594, 62.6144562
7: -66.8240051, -0.4220486, -66.9062042, -0.3443756, -55.6307106, 55.5888519
8: -42.7642555, 24.9166679, -42.8658485, 25.0149117, -67.7791672, 67.7825165
9: -21.6098595, 16.4572964, -21.6385345, 16.4818211, -38.0916824, 38.0958328
10: -53.1933746, 17.4246120, -53.2645874, 17.4660530, -68.5051727, 68.4670105
11: -69.5754547, -11.5942602, -69.6602783, -11.5908871, -46.6565018, 46.7148552
12: -32.5340691, 30.0383263, -32.5839272, 30.0869217, -59.7073593, 59.6943893
13: -35.8356476, 37.2986069, -35.8557892, 37.3656273, -73.2012787, 73.1543961
14: -105.3182831, -11.0636339, -105.4800034, -10.9911098, -89.0879517, 89.1284790
15: -35.2574844, 21.8787289, -35.2988853, 21.9023590, -56.7682877, 56.7237968
16: -60.9810028, 2.2741632, -61.0391655, 2.3048325, -54.8875504, 54.8641624
17: -122.9475708, -17.6890984, -123.0601120, -17.6397095, -82.4066849, 82.3032150
18: -46.8877487, 24.1254902, -46.9870834, 24.1890717, -71.0768204, 71.1125717
19: -40.2054977, 1.7214479, -40.2366180, 1.7430577, -39.6502037, 39.6397476
20: -31.6587315, 5.3753414, -31.6943836, 5.3881731, -37.0469055, 37.0697250
21: -53.1850624, 0.1933622, -53.2438431, 0.2089691, -49.8062439, 49.8171310
22: -53.9135284, 6.0764952, -53.9824677, 6.1176319, -55.7477722, 55.7622223
23: -32.7995491, 8.2645350, -32.8283920, 8.2731800, -38.6918488, 38.7374039
24: -25.9949951, 18.4603653, -26.0549603, 18.4874630, -43.8958778, 43.9305115
25: -23.3823128, 19.6765385, -23.4366741, 19.7336960, -41.6105652, 41.6105118
26: -44.8398628, 25.2245216, -44.9518509, 25.2655144, -70.0716400, 70.1484070
27: -45.6636581, 10.9494324, -45.7171097, 10.9620743, -56.6257324, 56.6665421
28: -36.1633224, 14.3717108, -36.1979065, 14.3889303, -48.7390137, 48.8463516
29: -65.2404938, -6.1600647, -65.3188629, -6.1369095, -50.1415787, 50.1323318
30: -43.7301216, 14.4108047, -43.7877121, 14.4381084, -58.1682281, 58.1985168
31: -41.5912056, 2.8551040, -41.6295624, 2.8820372, -44.4732437, 44.4846649
32: -38.7376404, 22.5201321, -38.7681808, 22.5564003, -61.2940407, 61.2883148
33: -19.6469212, 60.0168724, -19.7090340, 60.1241608, -73.4028625, 73.3994827
34: -28.1992149, 47.5488625, -28.2584686, 47.6147003, -70.3547134, 70.4180679
35: -18.5674782, 56.0478745, -18.5888443, 56.1058502, -71.4939880, 71.4892731
36: -27.2563972, 48.2715149, -27.3149643, 48.3411903, -73.6023560, 73.6285400
37: -14.6848831, 48.4097862, -14.7687187, 48.4748840, -54.7394791, 54.8239555
38: -33.2475510, 57.6852417, -33.2922668, 57.7486839, -87.4346390, 87.4462433
39: -19.6499043, 65.7978821, -19.6650105, 65.8818665, -77.2678375, 77.2663269
40: -22.8177147, 42.2931824, -22.8685303, 42.3470573, -61.6187019, 61.7115555
41: -26.0386467, 26.2900200, -26.0673027, 26.3089256, -52.3475723, 52.3573227
42: -35.6342773, 19.5555878, -35.6609802, 19.5828457, -55.2171249, 55.2165680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2540342, upper bound: 39.2427708
time: 59.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2547309
time: 54.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -63.4679031, 8.5782146, -63.4957199, 8.5963449, -72.0642471, 72.0739365
1: -44.6197510, 6.6942701, -44.6537933, 6.6688061, -48.3699837, 48.3822937
2: -35.3635101, 10.6874084, -35.4054642, 10.7120495, -43.5299187, 43.4928513
3: -46.0352478, 7.4451847, -46.0619659, 7.4693995, -50.0968170, 50.0600815
4: -36.2557030, 19.7113152, -36.2630005, 19.6566505, -54.6460571, 54.6339493
5: -49.2202911, 11.1180696, -49.2499619, 11.1289177, -54.9859161, 54.9364357
6: -43.6052475, 19.2217178, -43.5138435, 19.1922951, -62.7975426, 62.7355614
7: -66.8602905, -0.3459015, -66.9164581, -0.3328457, -55.7568626, 55.7504196
8: -42.8313255, 25.0114326, -42.8924026, 25.0284157, -67.8597412, 67.9038391
9: -21.6939697, 16.5432281, -21.6535282, 16.4881172, -38.1820869, 38.1967545
10: -53.3096733, 17.5959454, -53.3184776, 17.4795227, -68.6339417, 68.6991043
11: -69.6443863, -11.4807339, -69.6887207, -11.5860367, -46.7995872, 46.9364128
12: -32.6032181, 30.0743332, -32.6024895, 30.0933800, -59.8921852, 59.8553505
13: -36.0112457, 37.4183617, -35.8636322, 37.4217758, -73.4330215, 73.2819977
14: -105.5356598, -10.8601637, -105.5804977, -10.9865112, -89.2910385, 89.4396667
15: -35.3352852, 22.0131645, -35.3335419, 21.9115391, -56.8976059, 56.9418488
16: -61.0854874, 2.4129553, -61.0750809, 2.3193092, -54.9780579, 55.0175552
17: -123.1285477, -17.4642715, -123.1486740, -17.6262550, -82.4556274, 82.4994583
18: -47.0070038, 24.2828465, -47.0443497, 24.1950073, -71.2020111, 71.3271942
19: -40.2714195, 1.7824106, -40.2594452, 1.7500710, -39.6307793, 39.6354065
20: -31.7216206, 5.4243965, -31.7094231, 5.3905430, -37.1121635, 37.1338196
21: -53.2969513, 0.2914143, -53.2852020, 0.2128868, -49.8661728, 49.8997841
22: -54.0622101, 6.1785212, -54.0488663, 6.1219387, -55.7305527, 55.7710609
23: -32.8585472, 8.3353748, -32.8480186, 8.2770252, -38.7326736, 38.8086243
24: -26.1080360, 18.5499153, -26.1047897, 18.4898758, -44.0384598, 44.1030922
25: -23.4624462, 19.7418518, -23.4569530, 19.7362175, -41.6433640, 41.6486359
26: -44.9865112, 25.3737831, -45.0152359, 25.2701530, -70.2566681, 70.3890228
27: -45.7759933, 11.0512180, -45.7647934, 10.9655914, -56.7415848, 56.8160095
28: -36.2206802, 14.4217768, -36.2158813, 14.3930206, -48.7859192, 48.9037781
29: -65.3749542, -6.0475197, -65.3816605, -6.1324043, -50.2088699, 50.2501984
30: -43.8399048, 14.5240326, -43.8327560, 14.4427509, -58.2826538, 58.3567886
31: -41.6537170, 2.9278755, -41.6456566, 2.8876562, -44.5413742, 44.5735321
32: -38.8842087, 22.6197357, -38.7866135, 22.6049042, -61.4891129, 61.4063492
33: -19.8678131, 60.1205788, -19.7219391, 60.1778030, -73.6915436, 73.5238419
34: -28.3199806, 47.6195412, -28.2714958, 47.6441650, -70.5453873, 70.5335083
35: -18.7317009, 56.1405945, -18.5982819, 56.1523361, -71.6931229, 71.5738831
36: -27.4374371, 48.3903961, -27.3261299, 48.4049683, -73.8523254, 73.7559662
37: -14.8637886, 48.4536362, -14.7883759, 48.4972649, -54.8953705, 54.8266296
38: -33.4811172, 57.8406677, -33.3074112, 57.8226967, -87.7308273, 87.5953445
39: -19.8908997, 65.9186630, -19.6833992, 65.9482422, -77.5713577, 77.3889236
40: -22.9712200, 42.3781891, -22.8880901, 42.3901482, -61.8391838, 61.8349609
41: -26.1693306, 26.3787231, -26.0735149, 26.3479309, -52.5172615, 52.4522400
42: -35.7404747, 19.6410789, -35.6682281, 19.6206856, -55.3611603, 55.3093071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2539583, upper bound: 39.2579635
time: 58.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2425110, upper bound: 39.2698848
time: 64.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -63.4393692, 8.5747976, -63.3625107, 8.5131083, -71.9524765, 71.9373093
1: -44.5920029, 6.6579027, -44.5325241, 6.5817261, -48.2181396, 48.2680092
2: -35.3943291, 10.6885548, -35.3453674, 10.6295433, -43.3599586, 43.4226074
3: -46.1003876, 7.4478455, -46.0801620, 7.4209509, -49.9603882, 50.0197296
4: -36.1700096, 19.6236115, -36.1035957, 19.5275059, -54.3690834, 54.4217987
5: -49.2551117, 11.0910692, -49.2120972, 11.0430536, -54.7332077, 54.7605515
6: -43.5205841, 19.0339069, -43.4817810, 18.9911366, -62.5117188, 62.5156860
7: -66.9129181, -0.3489552, -66.8651886, -0.4105396, -55.5849380, 55.6779671
8: -42.8390427, 25.0000248, -42.7695618, 24.9385548, -67.7775955, 67.7695847
9: -21.5894985, 16.4768639, -21.5368176, 16.4259930, -38.0154915, 38.0136795
10: -53.1081467, 17.4504185, -53.0126572, 17.3404617, -68.2564545, 68.3099823
11: -69.6402893, -11.5805120, -69.6185303, -11.6349583, -46.6165543, 46.6947060
12: -32.5485954, 30.1021767, -32.5000496, 30.0692158, -59.6968002, 59.6836967
13: -35.8277702, 37.2604752, -35.7543716, 37.2164688, -73.0442352, 73.0148468
14: -105.3586273, -10.9873171, -105.2873154, -11.0815125, -88.8823853, 89.0523911
15: -35.2156372, 21.8933525, -35.1457291, 21.8297024, -56.6299210, 56.6569748
16: -60.9318123, 2.2924814, -60.8561287, 2.2317696, -54.7426605, 54.7500992
17: -122.8428650, -17.6468887, -122.7226944, -17.7947674, -82.1153030, 82.1861572
18: -46.9026642, 24.2327957, -46.8661194, 24.1843185, -71.0869827, 71.0989151
19: -40.1817551, 1.7347522, -40.1416969, 1.6898775, -39.5921135, 39.5868301
20: -31.6782913, 5.3776331, -31.6444702, 5.3413153, -37.0196075, 37.0221024
21: -53.1676331, 0.2014151, -53.1250191, 0.1407747, -49.7207451, 49.7436523
22: -53.8783531, 6.1470537, -53.8361816, 6.1057129, -55.7311859, 55.7045326
23: -32.7957230, 8.2613926, -32.7733154, 8.2309828, -38.7020950, 38.6636429
24: -25.9986305, 18.4940224, -25.9857025, 18.4564686, -43.8704491, 43.8932648
25: -23.4091949, 19.7808170, -23.3797455, 19.7570820, -41.6714096, 41.6169662
26: -44.8788452, 25.2898865, -44.8484573, 25.2472687, -70.0774994, 70.0937729
27: -45.6642456, 10.9416580, -45.6522713, 10.8941326, -56.5583801, 56.5939293
28: -36.1710892, 14.3682814, -36.1304398, 14.3326921, -48.7928925, 48.7381439
29: -65.2114410, -6.1403656, -65.1672745, -6.2080908, -50.0350227, 50.0870972
30: -43.7263489, 14.3753004, -43.6427002, 14.2920132, -58.0183640, 58.0180016
31: -41.5885925, 2.8739047, -41.5488739, 2.8075018, -44.3960953, 44.4227791
32: -38.7421112, 22.4546795, -38.6956024, 22.3994350, -61.1415482, 61.1502838
33: -19.6725750, 60.0112915, -19.5124283, 59.9279251, -73.2986526, 73.1280975
34: -28.2337799, 47.4885101, -28.0661011, 47.3685303, -70.2217712, 70.1347656
35: -18.5681190, 55.9512863, -18.4146805, 55.8514519, -71.2632294, 71.1957397
36: -27.3090401, 48.1715851, -27.1970024, 48.0889511, -73.4587402, 73.3937073
37: -14.7438354, 48.4442902, -14.6087208, 48.3991508, -54.7892609, 54.6094818
38: -33.2844734, 57.5587234, -33.1671371, 57.4623985, -87.2214813, 87.2165451
39: -19.6312504, 65.7269897, -19.5287285, 65.6529465, -77.0606613, 76.9887390
40: -22.8386917, 42.2250366, -22.7333641, 42.1575394, -61.5473671, 61.4992065
41: -26.0757980, 26.2301483, -26.0280418, 26.1939411, -52.2697372, 52.2581902
42: -35.6472626, 19.5199928, -35.6046753, 19.4933929, -55.1406555, 55.1246681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2082335, upper bound: 39.2698033
time: 69.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2339518, upper bound: 39.2817651
time: 54.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -63.4625473, 8.5805063, -63.4077034, 8.5409613, -72.0035095, 71.9882126
1: -44.6232300, 6.6641397, -44.5858154, 6.6245022, -48.2927818, 48.3054085
2: -35.4185257, 10.6954107, -35.3872070, 10.6682215, -43.4211998, 43.4524498
3: -46.1061554, 7.4520998, -46.0931778, 7.4350243, -49.9875107, 50.0265007
4: -36.1994858, 19.6307716, -36.1535301, 19.5903931, -54.4785614, 54.4296761
5: -49.2698669, 11.0991325, -49.2396431, 11.0779457, -54.7835236, 54.7790642
6: -43.5243988, 19.0436535, -43.4971924, 19.0140419, -62.5384407, 62.5408478
7: -66.9258347, -0.3446903, -66.8884430, -0.3787117, -55.6236191, 55.6930542
8: -42.8660278, 25.0062771, -42.8168564, 24.9688091, -67.8348389, 67.8231354
9: -21.6187859, 16.4790115, -21.5892410, 16.4460487, -38.0648346, 38.0682526
10: -53.1727715, 17.4575310, -53.1257019, 17.3991852, -68.3798828, 68.3794785
11: -69.6556168, -11.5759716, -69.6525192, -11.6120625, -46.6684952, 46.7475777
12: -32.5706635, 30.1112385, -32.5495071, 30.0980816, -59.7530975, 59.7207222
13: -35.8531990, 37.2650909, -35.8024597, 37.2351723, -73.0883713, 73.0675507
14: -105.3740768, -10.9833031, -105.3234940, -11.0475187, -88.9578018, 89.0854645
15: -35.2400627, 21.8995323, -35.1940384, 21.8549881, -56.6955147, 56.6792603
16: -60.9808807, 2.2966022, -60.9472389, 2.2501945, -54.8053131, 54.8295784
17: -122.9095306, -17.6433945, -122.8458405, -17.7119026, -82.2512512, 82.1763840
18: -46.9138870, 24.2477798, -46.8960037, 24.2116356, -71.1255188, 71.1437836
19: -40.2086411, 1.7380080, -40.1895676, 1.7210979, -39.6369438, 39.6155281
20: -31.6834316, 5.3978267, -31.6657772, 5.3858194, -37.0692520, 37.0636024
21: -53.1972771, 0.2088757, -53.1825562, 0.1791649, -49.7816505, 49.7908592
22: -53.8984146, 6.1518803, -53.8759232, 6.1292801, -55.7547989, 55.7366943
23: -32.8028488, 8.2764330, -32.7933578, 8.2592793, -38.7460480, 38.7084465
24: -26.0018234, 18.5188465, -26.0086346, 18.5008621, -43.9166985, 43.9425812
25: -23.4142208, 19.7950516, -23.3979168, 19.7842560, -41.7009621, 41.6494293
26: -44.8876038, 25.3017235, -44.8719254, 25.2716236, -70.1054459, 70.1459045
27: -45.6692657, 10.9667034, -45.6715469, 10.9439735, -56.6132393, 56.6382523
28: -36.1759377, 14.3988628, -36.1709976, 14.3851557, -48.8080406, 48.8371620
29: -65.2388535, -6.1341982, -65.2171478, -6.1685734, -50.0826988, 50.0939865
30: -43.7364731, 14.4555120, -43.7418060, 14.4323788, -58.1688538, 58.1973190
31: -41.6170998, 2.8820844, -41.5999756, 2.8600888, -44.4771881, 44.4820595
32: -38.7538414, 22.4639854, -38.7215195, 22.4277687, -61.1816101, 61.1855049
33: -19.6980782, 60.0695763, -19.6257553, 60.0258408, -73.4018936, 73.2985687
34: -28.2509995, 47.5872841, -28.2106819, 47.5354843, -70.3653412, 70.3734283
35: -18.5834846, 56.0252762, -18.5278625, 55.9750824, -71.3642654, 71.3674774
36: -27.3130608, 48.2076988, -27.2491722, 48.1483459, -73.4882660, 73.4669418
37: -14.7638283, 48.4796638, -14.7091103, 48.4577942, -54.8319206, 54.7413788
38: -33.2871323, 57.5886993, -33.2069321, 57.5211296, -87.2743759, 87.2450485
39: -19.6518440, 65.7498322, -19.5748482, 65.6954422, -77.1084595, 77.0686035
40: -22.8583641, 42.2702751, -22.8219376, 42.2343102, -61.5855408, 61.6313667
41: -26.0849667, 26.2361565, -26.0543575, 26.2107716, -52.2957382, 52.2905121
42: -35.6643791, 19.5247040, -35.6419907, 19.5091209, -55.1735001, 55.1666946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2425891, upper bound: 39.2698033
time: 54.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2683065, upper bound: 39.2817651
time: 61.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -63.5283241, 8.6440392, -63.3876686, 8.5226851, -72.0510101, 72.0317078
1: -44.6504326, 6.7643738, -44.5557289, 6.5880957, -48.2898331, 48.4095573
2: -35.4122849, 10.7328300, -35.3488007, 10.6396179, -43.4600983, 43.5325623
3: -46.1358032, 7.4976540, -46.0849228, 7.4339724, -50.1199532, 50.1598167
4: -36.2394180, 19.7359467, -36.1279068, 19.5403595, -54.4917526, 54.6040497
5: -49.2848473, 11.1484289, -49.2174301, 11.0589209, -54.9639969, 54.9805260
6: -43.6468201, 19.1503239, -43.4864120, 19.0454979, -62.6923180, 62.6367340
7: -66.9502487, -0.2723103, -66.8761292, -0.3987942, -55.7106743, 55.8391876
8: -42.9060020, 25.0953217, -42.7959061, 24.9523125, -67.8583145, 67.8912277
9: -21.6737041, 16.5628223, -21.5519581, 16.4323139, -38.1060181, 38.1147804
10: -53.2244797, 17.6218262, -53.0665054, 17.3540802, -68.3854828, 68.5421600
11: -69.7092285, -11.4668798, -69.6469269, -11.6300993, -46.7602272, 46.9162064
12: -32.6177216, 30.1382027, -32.5188065, 30.0755348, -59.8811646, 59.8448792
13: -36.0033417, 37.3801651, -35.7622452, 37.2725906, -73.2759323, 73.1424103
14: -105.5759888, -10.7839317, -105.3876953, -11.0769110, -89.0854568, 89.3635254
15: -35.2934418, 22.0281029, -35.1802902, 21.8389740, -56.7589531, 56.8751411
16: -61.0363503, 2.4313221, -60.8920593, 2.2464056, -54.8330612, 54.9031982
17: -123.0238876, -17.4219742, -122.8112946, -17.7811966, -82.1644363, 82.3823624
18: -47.0218277, 24.3902054, -46.9233475, 24.1902657, -71.2120972, 71.3135529
19: -40.2476883, 1.7957473, -40.1645088, 1.6969419, -39.5730934, 39.5825806
20: -31.7411118, 5.4266624, -31.6596146, 5.3436661, -37.0847778, 37.0862770
21: -53.2795334, 0.2994757, -53.1663589, 0.1446562, -49.7808990, 49.8261871
22: -54.0270081, 6.2490826, -53.9026871, 6.1100311, -55.7139435, 55.7136345
23: -32.8545914, 8.3322735, -32.7927856, 8.2348537, -38.7427292, 38.7348404
24: -26.1115608, 18.5835419, -26.0355434, 18.4588890, -44.0128784, 44.0657959
25: -23.4890862, 19.8461189, -23.4009686, 19.7595997, -41.7041550, 41.6551323
26: -45.0254936, 25.4391708, -44.9117966, 25.2519226, -70.2774200, 70.3509674
27: -45.7764816, 11.0434303, -45.6999893, 10.8976803, -56.6741638, 56.7434196
28: -36.2283630, 14.4183550, -36.1483116, 14.3367996, -48.8396759, 48.7956123
29: -65.3458862, -6.0278177, -65.2303009, -6.2035980, -50.1022758, 50.2050705
30: -43.8360672, 14.4885187, -43.6875610, 14.2966900, -58.1327591, 58.1760788
31: -41.6511421, 2.9467788, -41.5649643, 2.8131509, -44.4642944, 44.5117416
32: -38.8890038, 22.5540771, -38.7142258, 22.4477310, -61.3367348, 61.2683029
33: -19.8934402, 60.1149750, -19.5253792, 59.9815598, -73.5872803, 73.2524719
34: -28.3545494, 47.5591393, -28.0792198, 47.3979340, -70.4123306, 70.2501297
35: -18.7323952, 56.0439911, -18.4241753, 55.8979340, -71.4623642, 71.2804108
36: -27.4901600, 48.2904587, -27.2082806, 48.1526527, -73.7087250, 73.5212936
37: -14.9227982, 48.4881325, -14.6283951, 48.4215317, -54.9452362, 54.6122284
38: -33.5182724, 57.7141953, -33.1823845, 57.5364380, -87.5176086, 87.3652878
39: -19.8722763, 65.8477020, -19.5471916, 65.7193298, -77.3639450, 77.1113739
40: -22.9922218, 42.3099632, -22.7530022, 42.2005692, -61.7671280, 61.6222649
41: -26.2065048, 26.3186398, -26.0342579, 26.2328281, -52.4393311, 52.3528976
42: -35.7536469, 19.6054401, -35.6118050, 19.5310764, -55.2847214, 55.2172470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2081527, upper bound: 39.2850114
time: 58.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2339518, upper bound: 39.2969124
time: 59.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -63.5514717, 8.6497574, -63.4328690, 8.5505486, -72.1020203, 72.0826263
1: -44.6816216, 6.7706261, -44.6089973, 6.6308517, -48.3644714, 48.4469643
2: -35.4364967, 10.7396908, -35.3906326, 10.6782780, -43.5213623, 43.5623703
3: -46.1415672, 7.5018749, -46.0979309, 7.4480753, -50.1470451, 50.1665649
4: -36.2688293, 19.7430992, -36.1777954, 19.6032677, -54.6011963, 54.6119347
5: -49.2995911, 11.1564665, -49.2449799, 11.0938292, -55.0143051, 54.9990463
6: -43.6506348, 19.1600494, -43.5018234, 19.0684376, -62.7190704, 62.6618729
7: -66.9631653, -0.2680492, -66.8993835, -0.3669624, -55.7493439, 55.8542557
8: -42.9329453, 25.1015720, -42.8432045, 24.9825459, -67.9154892, 67.9447784
9: -21.7029724, 16.5649738, -21.6043186, 16.4523735, -38.1553459, 38.1692924
10: -53.2891121, 17.6289482, -53.1795273, 17.4128113, -68.5089264, 68.6116333
11: -69.7245483, -11.4623404, -69.6808853, -11.6071701, -46.8121719, 46.9691353
12: -32.6397781, 30.1472664, -32.5683022, 30.1044350, -59.9375000, 59.8818779
13: -36.0288010, 37.3847504, -35.8103561, 37.2912483, -73.3200531, 73.1951065
14: -105.5913696, -10.7798948, -105.4239044, -11.0429001, -89.1607971, 89.3965454
15: -35.3178406, 22.0342865, -35.2285995, 21.8642845, -56.8245163, 56.8974152
16: -61.0854340, 2.4354429, -60.9831314, 2.2648087, -54.8959618, 54.9829140
17: -123.0905609, -17.4184914, -122.9344025, -17.6982975, -82.3003235, 82.3726044
18: -47.0330505, 24.4051895, -46.9532623, 24.2175808, -71.2506332, 71.3584518
19: -40.2745743, 1.7989883, -40.2123795, 1.7281427, -39.6179314, 39.6112862
20: -31.7462349, 5.4468703, -31.6809235, 5.3881721, -37.1344070, 37.1277924
21: -53.3091354, 0.3069458, -53.2238655, 0.1830435, -49.8417740, 49.8733826
22: -54.0470428, 6.2539043, -53.9424515, 6.1335802, -55.7375565, 55.7457733
23: -32.8617020, 8.3473244, -32.8128357, 8.2631636, -38.7867470, 38.7796364
24: -26.1147194, 18.6083908, -26.0584602, 18.5032845, -44.0591507, 44.1151237
25: -23.4940987, 19.8603725, -23.4191532, 19.7867737, -41.7336349, 41.6875916
26: -45.0342789, 25.4510193, -44.9352493, 25.2762642, -70.3105469, 70.3862686
27: -45.7814560, 11.0684853, -45.7192612, 10.9475174, -56.7289734, 56.7877464
28: -36.2331848, 14.4489498, -36.1888580, 14.3892584, -48.8548279, 48.8946075
29: -65.3732910, -6.0216303, -65.2801819, -6.1640682, -50.1498489, 50.2119102
30: -43.8461533, 14.5687466, -43.7866402, 14.4370670, -58.2832184, 58.3553848
31: -41.6796646, 2.9549408, -41.6160736, 2.8657503, -44.5454140, 44.5710144
32: -38.9007568, 22.5633545, -38.7401810, 22.4760513, -61.3768082, 61.3035355
33: -19.9189453, 60.1732712, -19.6387367, 60.0794907, -73.6905975, 73.4229584
34: -28.3717957, 47.6579056, -28.2238522, 47.5649033, -70.5559464, 70.4888306
35: -18.7477570, 56.1179657, -18.5373840, 56.0215912, -71.5634155, 71.4521561
36: -27.4941940, 48.3265877, -27.2604294, 48.2120361, -73.7382812, 73.5944824
37: -14.9428158, 48.5235062, -14.7287750, 48.4801674, -54.9878616, 54.7440758
38: -33.5209274, 57.7441216, -33.2221642, 57.5951195, -87.5704498, 87.3938751
39: -19.8928585, 65.8705597, -19.5933342, 65.7617950, -77.4119568, 77.1912079
40: -23.0119495, 42.3551941, -22.8415718, 42.2773476, -61.8053474, 61.7543602
41: -26.2157555, 26.3246479, -26.0605698, 26.2496681, -52.4654236, 52.3852158
42: -35.7707977, 19.6101131, -35.6491470, 19.5468140, -55.3176117, 55.2592621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2425110, upper bound: 39.2850115
time: 61.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2683065, upper bound: 39.2969124
time: 56.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -63.5193481, 8.5838633, -63.5178261, 8.5693703, -72.0887146, 72.1016922
1: -44.6512947, 6.6662302, -44.6460762, 6.6342506, -48.3297462, 48.3680534
2: -35.4382591, 10.6988773, -35.4275055, 10.6748991, -43.4490891, 43.4869957
3: -46.1184578, 7.4603024, -46.1161194, 7.4580517, -50.0120697, 50.0539818
4: -36.2369576, 19.6375561, -36.2332230, 19.5913658, -54.5009384, 54.5588531
5: -49.2919960, 11.1048203, -49.2821274, 11.0899897, -54.8119736, 54.8154526
6: -43.5275154, 19.1095829, -43.5271263, 19.1348324, -62.6623459, 62.6367111
7: -66.9719238, -0.3391838, -66.9769363, -0.3578892, -55.6997757, 55.7619209
8: -42.9184570, 25.0158119, -42.9206505, 25.0024815, -67.9209366, 67.9364624
9: -21.6211510, 16.4853306, -21.6074886, 16.4730511, -38.0942001, 38.0928192
10: -53.2020454, 17.4657326, -53.1902771, 17.4246120, -68.4343567, 68.4876175
11: -69.6801682, -11.5741386, -69.6959610, -11.5944128, -46.6959610, 46.7433891
12: -32.5667686, 30.1222610, -32.5479736, 30.1084003, -59.7636375, 59.7658157
13: -35.8404541, 37.3385773, -35.8200722, 37.3649292, -73.2053833, 73.1586456
14: -105.4983139, -10.9781256, -105.5515900, -11.0028143, -89.1067581, 89.2991638
15: -35.2851524, 21.9040337, -35.2798843, 21.8932114, -56.7634621, 56.7932701
16: -61.0005264, 2.3076229, -60.9885063, 2.3036156, -54.8826485, 54.8585396
17: -122.9882889, -17.6358700, -122.9969559, -17.7049961, -82.3505478, 82.3877411
18: -46.9625320, 24.2428455, -46.9808197, 24.2352695, -71.1977997, 71.2236633
19: -40.2121964, 1.7405267, -40.2058792, 1.7180233, -39.6458664, 39.6301460
20: -31.6929855, 5.3831816, -31.6883144, 5.3593631, -37.0523491, 37.0714951
21: -53.2102699, 0.2095013, -53.2146683, 0.1822786, -49.8015442, 49.8173752
22: -53.9452553, 6.1536045, -53.9678917, 6.1439781, -55.8361969, 55.8127174
23: -32.8165398, 8.2679358, -32.8193359, 8.2539244, -38.7452087, 38.7154732
24: -26.0274849, 18.4977188, -26.0453396, 18.4777565, -43.9257317, 43.9565773
25: -23.4274349, 19.7861710, -23.4308147, 19.7779522, -41.7120209, 41.6763496
26: -44.9311333, 25.2982178, -44.9564972, 25.2900429, -70.1907959, 70.2099762
27: -45.6962433, 10.9482555, -45.7169113, 10.9238796, -56.6201248, 56.6651688
28: -36.1878281, 14.3758020, -36.1662941, 14.3530588, -48.8320007, 48.7910309
29: -65.2851028, -6.1353207, -65.3069611, -6.1648979, -50.1521835, 50.1892433
30: -43.7536240, 14.3836823, -43.6992722, 14.3245544, -58.0781784, 58.0829544
31: -41.6039810, 2.8808851, -41.5891991, 2.8404717, -44.4444542, 44.4700851
32: -38.7576981, 22.5305176, -38.7597694, 22.5421295, -61.2998276, 61.2902870
33: -19.6928978, 60.1124382, -19.6148605, 60.1153412, -73.4841003, 73.3309937
34: -28.2515354, 47.5646744, -28.1356010, 47.5119095, -70.3506699, 70.2802811
35: -18.5813828, 56.0465927, -18.4908409, 56.0280533, -71.4315720, 71.3657074
36: -27.3222065, 48.2943840, -27.2823334, 48.3150291, -73.6802521, 73.6022491
37: -14.7668152, 48.4902115, -14.7013779, 48.4840164, -54.8686829, 54.7498055
38: -33.3047791, 57.6934204, -33.2801170, 57.7112465, -87.4550171, 87.4631119
39: -19.6518288, 65.8582001, -19.6361160, 65.8916779, -77.2871857, 77.2274780
40: -22.8574905, 42.3007507, -22.8036976, 42.2989883, -61.6673584, 61.6441040
41: -26.0863495, 26.2876358, -26.0792236, 26.3038387, -52.3901901, 52.3668594
42: -35.6582108, 19.5670204, -35.6492386, 19.5836124, -55.2418213, 55.2162590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2196852, upper bound: 39.2698033
time: 54.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2625730, upper bound: 39.2817651
time: 76.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -63.5425186, 8.5895948, -63.5630150, 8.5972099, -72.1397247, 72.1526108
1: -44.6825409, 6.6725016, -44.6993752, 6.6770077, -48.4043808, 48.4054222
2: -35.4624405, 10.7057371, -35.4693146, 10.7135553, -43.5103416, 43.5168076
3: -46.1242180, 7.4645519, -46.1291199, 7.4721260, -50.0392303, 50.0607376
4: -36.2664528, 19.6447182, -36.2831345, 19.6542625, -54.6104164, 54.5666885
5: -49.3067322, 11.1129284, -49.3096848, 11.1248941, -54.8623657, 54.8339500
6: -43.5313416, 19.1193237, -43.5425339, 19.1577282, -62.6890717, 62.6618576
7: -66.9848175, -0.3349190, -67.0001755, -0.3260803, -55.7384415, 55.7769661
8: -42.9454422, 25.0220661, -42.9679489, 25.0326881, -67.9781342, 67.9900131
9: -21.6504211, 16.4874725, -21.6598682, 16.4930992, -38.1435204, 38.1473389
10: -53.2666740, 17.4728355, -53.3033104, 17.4833450, -68.5577850, 68.5570526
11: -69.6954727, -11.5695572, -69.7299194, -11.5714750, -46.7479248, 46.7962685
12: -32.5888252, 30.1313133, -32.5975037, 30.1372910, -59.8199997, 59.8028831
13: -35.8659515, 37.3431816, -35.8682404, 37.3835487, -73.2494965, 73.2114258
14: -105.5137482, -10.9741135, -105.5877762, -10.9688072, -89.1821899, 89.3322144
15: -35.3095512, 21.9102211, -35.3281708, 21.9185066, -56.8290558, 56.8155518
16: -61.0495796, 2.3117552, -61.0795822, 2.3220415, -54.9452744, 54.9379921
17: -123.0550156, -17.6324310, -123.1201630, -17.6220932, -82.4864655, 82.3779755
18: -46.9737625, 24.2578259, -47.0107269, 24.2625999, -71.2363586, 71.2685547
19: -40.2390785, 1.7437758, -40.2537384, 1.7492218, -39.6906891, 39.6588364
20: -31.6981106, 5.4033761, -31.7096272, 5.4038639, -37.1019745, 37.1130028
21: -53.2398872, 0.2169752, -53.2721748, 0.2206764, -49.8624077, 49.8645592
22: -53.9653244, 6.1584435, -54.0076065, 6.1675558, -55.8598518, 55.8448296
23: -32.8236771, 8.2829885, -32.8393784, 8.2822428, -38.7891541, 38.7602615
24: -26.0306644, 18.5225410, -26.0682602, 18.5221672, -43.9720306, 44.0058708
25: -23.4324703, 19.8004036, -23.4489841, 19.8051624, -41.7415848, 41.7088280
26: -44.9399109, 25.3100815, -44.9799728, 25.3144341, -70.2187500, 70.2621460
27: -45.7012558, 10.9733038, -45.7362099, 10.9737349, -56.6749916, 56.7095146
28: -36.1926651, 14.4063921, -36.2068443, 14.4055138, -48.8471489, 48.8900566
29: -65.3125305, -6.1291113, -65.3568726, -6.1253777, -50.1998672, 50.1961250
30: -43.7637482, 14.4639034, -43.7983704, 14.4649258, -58.2286758, 58.2622757
31: -41.6324959, 2.8890700, -41.6402817, 2.8930578, -44.5255547, 44.5293503
32: -38.7694435, 22.5397987, -38.7857170, 22.5704308, -61.3398743, 61.3255157
33: -19.7184105, 60.1707382, -19.7281685, 60.2132568, -73.5873718, 73.5014801
34: -28.2687607, 47.6634369, -28.2802048, 47.6788635, -70.4942245, 70.5189514
35: -18.5967007, 56.1205597, -18.6040382, 56.1517487, -71.5325928, 71.5374451
36: -27.3262367, 48.3305054, -27.3344746, 48.3744316, -73.7097778, 73.6754608
37: -14.7868347, 48.5255661, -14.8018227, 48.5426483, -54.9113388, 54.8817329
38: -33.3074112, 57.7234039, -33.3199005, 57.7699509, -87.5079422, 87.4916077
39: -19.6724491, 65.8810883, -19.6822357, 65.9341583, -77.3349762, 77.3073425
40: -22.8771935, 42.3459854, -22.8923016, 42.3757629, -61.7055588, 61.7763481
41: -26.0955219, 26.2936478, -26.1055298, 26.3206673, -52.4161911, 52.3991776
42: -35.6753311, 19.5717258, -35.6866074, 19.5993309, -55.2746620, 55.2583313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2540342, upper bound: 39.2698033
time: 66.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2817651
time: 79.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -63.6076889, 8.6529589, -63.5426788, 8.5788584, -72.1865463, 72.1956406
1: -44.7099152, 6.7725048, -44.6695023, 6.6404934, -48.4015236, 48.5094833
2: -35.4557190, 10.7430840, -35.4302826, 10.6848955, -43.5487442, 43.5969009
3: -46.1536942, 7.5100880, -46.1210556, 7.4710302, -50.1715889, 50.1939087
4: -36.3064651, 19.7496128, -36.2577209, 19.6040363, -54.6238403, 54.7411995
5: -49.3215256, 11.1622000, -49.2875290, 11.1057024, -55.0399590, 55.0349579
6: -43.6537094, 19.2261181, -43.5317230, 19.1893196, -62.8430290, 62.7578430
7: -67.0081635, -0.2629833, -66.9872284, -0.3463688, -55.8258972, 55.9235153
8: -42.9855423, 25.1106186, -42.9472122, 25.0159950, -68.0015411, 68.0578308
9: -21.7052536, 16.5712471, -21.6225491, 16.4793377, -38.1845932, 38.1937943
10: -53.3183022, 17.6370621, -53.2442017, 17.4380779, -68.5631256, 68.7197266
11: -69.7490845, -11.4605398, -69.7244110, -11.5895596, -46.8390312, 46.9649277
12: -32.6359329, 30.1582642, -32.5665894, 30.1148529, -59.9484940, 59.9268761
13: -36.0159760, 37.4583130, -35.8278885, 37.4210587, -73.4370346, 73.2862015
14: -105.7156601, -10.7747288, -105.6520767, -10.9981918, -89.3098755, 89.6103745
15: -35.3630180, 22.0384674, -35.3145447, 21.9024086, -56.8928680, 57.0113602
16: -61.1050034, 2.4464006, -61.0244560, 2.3180904, -54.9729080, 55.0116882
17: -123.1692886, -17.4110489, -123.0855408, -17.6915989, -82.3995514, 82.5839539
18: -47.0817299, 24.4001980, -47.0380707, 24.2411861, -71.3229141, 71.4382706
19: -40.2780876, 1.8014927, -40.2287140, 1.7250228, -39.6264572, 39.6258278
20: -31.7559204, 5.4322157, -31.7033463, 5.3617373, -37.1176567, 37.1355629
21: -53.3221664, 0.3075447, -53.2560501, 0.1861725, -49.8614731, 49.9000626
22: -54.0939789, 6.2556276, -54.0342712, 6.1483135, -55.8190002, 55.8215370
23: -32.8755417, 8.3387814, -32.8389664, 8.2577667, -38.7859879, 38.7866592
24: -26.1405315, 18.5872192, -26.0951653, 18.4801750, -44.0683060, 44.1291084
25: -23.5075836, 19.8514748, -23.4510765, 19.7804794, -41.7448959, 41.7145157
26: -45.0778351, 25.4474888, -45.0199127, 25.2946815, -70.3725128, 70.4673996
27: -45.8085823, 11.0499878, -45.7646103, 10.9273949, -56.7359772, 56.8145981
28: -36.2452087, 14.4258690, -36.1842957, 14.3571558, -48.8788910, 48.8484573
29: -65.4195709, -6.0227690, -65.3698120, -6.1603832, -50.2194786, 50.3070221
30: -43.8634300, 14.4968872, -43.7443199, 14.3292122, -58.1926422, 58.2412071
31: -41.6665192, 2.9537067, -41.6053047, 2.8460732, -44.5125923, 44.5590134
32: -38.9042854, 22.6300907, -38.7781830, 22.5906487, -61.4949341, 61.4082718
33: -19.9137688, 60.2161446, -19.6277542, 60.1689529, -73.7727814, 73.4553223
34: -28.3722687, 47.6353226, -28.1486168, 47.5413589, -70.5412979, 70.3957062
35: -18.7455673, 56.1392555, -18.5002689, 56.0745506, -71.6306610, 71.4502792
36: -27.5032654, 48.4132652, -27.2935143, 48.3787918, -73.9302139, 73.7296448
37: -14.9456863, 48.5340500, -14.7210369, 48.5064125, -55.0245514, 54.7525368
38: -33.5383492, 57.8488655, -33.2952652, 57.7853012, -87.7512436, 87.6121368
39: -19.8928108, 65.9789734, -19.6544533, 65.9580536, -77.5905151, 77.3500977
40: -23.0108700, 42.3857193, -22.8232269, 42.3420792, -61.8878326, 61.7675972
41: -26.2169724, 26.3762989, -26.0854301, 26.3428307, -52.5598030, 52.4617310
42: -35.7643661, 19.6525402, -35.6564713, 19.6214676, -55.3858337, 55.3090134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1515

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2196196, upper bound: 39.2850115
time: 68.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2625730, upper bound: 39.2969124
time: 58.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -63.6308441, 8.6586838, -63.5878181, 8.6067142, -72.2375565, 72.2465057
1: -44.7411079, 6.7787628, -44.7227516, 6.6832590, -48.4761696, 48.5468788
2: -35.4799309, 10.7499428, -35.4720917, 10.7235565, -43.6099854, 43.6267128
3: -46.1594543, 7.5143089, -46.1340637, 7.4851322, -50.1986923, 50.2006950
4: -36.3358879, 19.7567806, -36.3075638, 19.6669369, -54.7332535, 54.7490463
5: -49.3362732, 11.1702490, -49.3150864, 11.1406279, -55.0903587, 55.0534668
6: -43.6575241, 19.2358208, -43.5471420, 19.2122173, -62.8697433, 62.7829628
7: -67.0210648, -0.2587109, -67.0104675, -0.3145370, -55.8645935, 55.9385910
8: -43.0124741, 25.1168613, -42.9944687, 25.0462036, -68.0586777, 68.1113281
9: -21.7344933, 16.5733929, -21.6748447, 16.4993935, -38.2338867, 38.2482376
10: -53.3829041, 17.6441765, -53.3571892, 17.4967957, -68.6865463, 68.7891846
11: -69.7644043, -11.4560099, -69.7583542, -11.5666466, -46.8910103, 47.0178375
12: -32.6579704, 30.1673336, -32.6161079, 30.1437454, -60.0048065, 59.9638901
13: -36.0414772, 37.4628830, -35.8760681, 37.4396973, -73.4811707, 73.3389511
14: -105.7310715, -10.7706928, -105.6882553, -10.9642258, -89.3852463, 89.6433640
15: -35.3874283, 22.0446739, -35.3628159, 21.9276886, -56.9584732, 57.0336189
16: -61.1540527, 2.4505529, -61.1154976, 2.3364973, -55.0357742, 55.0913582
17: -123.2359467, -17.4075851, -123.2086716, -17.6087112, -82.5354462, 82.5742035
18: -47.0929413, 24.4152012, -47.0679588, 24.2685242, -71.3614655, 71.4831619
19: -40.3049736, 1.8047323, -40.2765808, 1.7562385, -39.6712837, 39.6545410
20: -31.7610512, 5.4524212, -31.7246609, 5.4062309, -37.1672821, 37.1770821
21: -53.3517914, 0.3150263, -53.3135452, 0.2245770, -49.9223595, 49.9472313
22: -54.1140366, 6.2604628, -54.0740013, 6.1718969, -55.8426437, 55.8536797
23: -32.8826637, 8.3538361, -32.8590050, 8.2860985, -38.8299675, 38.8314743
24: -26.1436901, 18.6120605, -26.1180820, 18.5245781, -44.1146011, 44.1784515
25: -23.5126057, 19.8657341, -23.4692459, 19.8076897, -41.7743835, 41.7469635
26: -45.0866051, 25.4593353, -45.0433426, 25.3190670, -70.4056702, 70.5026779
27: -45.8135757, 11.0750599, -45.7838898, 10.9772472, -56.7908249, 56.8589478
28: -36.2500267, 14.4564533, -36.2248459, 14.4096031, -48.8940430, 48.9474754
29: -65.4469910, -6.0165920, -65.4196777, -6.1208849, -50.2670631, 50.3138733
30: -43.8735313, 14.5770988, -43.8434105, 14.4696064, -58.3431396, 58.4205093
31: -41.6950378, 2.9618788, -41.6563835, 2.8986778, -44.5937157, 44.6182632
32: -38.9160538, 22.6393585, -38.8041611, 22.6189499, -61.5350037, 61.4435196
33: -19.9393005, 60.2744446, -19.7410889, 60.2669182, -73.8760834, 73.6258163
34: -28.3895073, 47.7340622, -28.2932148, 47.7082977, -70.6848984, 70.6343536
35: -18.7609463, 56.2132301, -18.6134758, 56.1982193, -71.7317200, 71.6221008
36: -27.5072899, 48.4493790, -27.3456459, 48.4381714, -73.9597626, 73.8028412
37: -14.9657555, 48.5694084, -14.8214607, 48.5650444, -55.0672188, 54.8844261
38: -33.5410194, 57.8787956, -33.3350143, 57.8439789, -87.8040848, 87.6406860
39: -19.9133930, 66.0018387, -19.7006035, 66.0005569, -77.6384735, 77.4299622
40: -23.0306282, 42.4309692, -22.9118595, 42.4188461, -61.9260559, 61.8997536
41: -26.2262077, 26.3822918, -26.1117592, 26.3596497, -52.5858574, 52.4940491
42: -35.7815285, 19.6572266, -35.6938438, 19.6371765, -55.4187050, 55.3510704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=357, inp2_unstable=357, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1432

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2539583, upper bound: 39.2850115
time: 65.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2969124
time: 89.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 156.86 seconds
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2540342, upper bound: 39.2427708
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2547309
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2539583, upper bound: 39.2579635
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2425110, upper bound: 39.2698848
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2082335, upper bound: 39.2698033
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2339518, upper bound: 39.2817651
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2425891, upper bound: 39.2698033
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2683065, upper bound: 39.2817651
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2081527, upper bound: 39.2850114
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2339518, upper bound: 39.2969124
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2425110, upper bound: 39.2850115
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2683065, upper bound: 39.2969124
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2196852, upper bound: 39.2698033
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2625730, upper bound: 39.2817651
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2540342, upper bound: 39.2698033
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2817651
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2196196, upper bound: 39.2850115
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2625730, upper bound: 39.2969124
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2539583, upper bound: 39.2850115
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 156.86
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2969124

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -63.4431152, 8.5088730, -63.4674606, 8.5828323, -72.0259476, 71.9763336
1: -44.6167145, 6.5907202, -44.6291618, 6.6590462, -48.3500443, 48.2378731
2: -35.3692665, 10.6480837, -35.4016495, 10.6992731, -43.4487839, 43.3838959
3: -45.9997978, 7.4033098, -46.0519180, 7.4547071, -49.9340858, 49.9255638
4: -36.2213783, 19.6031895, -36.2370911, 19.6407967, -54.5570908, 54.4508018
5: -49.1960678, 11.0664310, -49.2422180, 11.1101837, -54.7571373, 54.7176285
6: -43.4811172, 19.1715794, -43.5059433, 19.1348991, -62.6160164, 62.6775208
7: -66.8572083, -0.4208679, -66.9043427, -0.3481884, -55.6594009, 55.5844917
8: -42.8306885, 24.9259281, -42.8644333, 25.0108624, -67.8415527, 67.7903595
9: -21.6397572, 16.4606247, -21.6365337, 16.4787216, -38.1184769, 38.0971603
10: -53.2933121, 17.4291344, -53.2629356, 17.4599419, -68.5987396, 68.4652557
11: -69.6296844, -11.5936270, -69.6578827, -11.5940914, -46.7091484, 46.7042618
12: -32.5461769, 30.0389862, -32.5821266, 30.0789280, -59.7135620, 59.6957970
13: -35.8411255, 37.4008102, -35.8509064, 37.3632736, -73.2043991, 73.2517166
14: -105.4594955, -11.0615826, -105.4750671, -10.9964409, -89.2282104, 89.1169357
15: -35.3116112, 21.8817863, -35.2972641, 21.8986855, -56.8214035, 56.7241211
16: -61.0397720, 2.2797585, -61.0365219, 2.3005209, -54.9414597, 54.8595772
17: -123.0733185, -17.6890907, -123.0571747, -17.6453800, -82.5255203, 82.2835846
18: -46.9642143, 24.1295681, -46.9846611, 24.1855927, -71.1498108, 71.1142273
19: -40.2287216, 1.7231627, -40.2347221, 1.7412877, -39.6710014, 39.6344986
20: -31.6665916, 5.3796549, -31.6919880, 5.3870840, -37.0536766, 37.0716438
21: -53.2252998, 0.1976576, -53.2402382, 0.2064104, -49.8392639, 49.8130302
22: -53.9560547, 6.0794725, -53.9784393, 6.1158266, -55.7872238, 55.7565804
23: -32.8271446, 8.2668514, -32.8260651, 8.2700672, -38.7074776, 38.7326393
24: -26.0173416, 18.4605198, -26.0516529, 18.4857063, -43.9169617, 43.9260902
25: -23.3927364, 19.6803436, -23.4341068, 19.7316246, -41.6187477, 41.6111412
26: -44.9117699, 25.2277184, -44.9486961, 25.2620525, -70.1394806, 70.1473618
27: -45.6932678, 10.9524164, -45.7132645, 10.9601183, -56.6533852, 56.6656799
28: -36.1729965, 14.3760490, -36.1956062, 14.3864956, -48.7433586, 48.8466530
29: -65.2948303, -6.1590891, -65.3144455, -6.1394653, -50.1903839, 50.1195755
30: -43.7566681, 14.4158792, -43.7841263, 14.4353333, -58.1920013, 58.2000046
31: -41.5998039, 2.8580647, -41.6250076, 2.8801689, -44.4799728, 44.4830704
32: -38.7427673, 22.5887775, -38.7624855, 22.5539513, -61.2967186, 61.3512650
33: -19.6575165, 60.1254387, -19.7030334, 60.1225700, -73.4064102, 73.5017624
34: -28.2079964, 47.6193581, -28.2540703, 47.6130753, -70.3564301, 70.4854279
35: -18.5714645, 56.1533966, -18.5833664, 56.1049156, -71.4921875, 71.5888977
36: -27.2592278, 48.3959961, -27.3087158, 48.3398743, -73.6001511, 73.7469254
37: -14.7002392, 48.4431610, -14.7632942, 48.4741516, -54.7487068, 54.8558006
38: -33.2570648, 57.7990685, -33.2857933, 57.7460709, -87.4341278, 87.5561523
39: -19.6585007, 65.9424820, -19.6577835, 65.8799515, -77.2645721, 77.4047852
40: -22.8313656, 42.3604126, -22.8634071, 42.3451881, -61.6226196, 61.7704506
41: -26.0436249, 26.3338757, -26.0636444, 26.3066673, -52.3502922, 52.3975220
42: -35.6402664, 19.5810242, -35.6586838, 19.5805740, -55.2208405, 55.2397079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1515

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2167188, upper bound: 39.2204211
time: 74.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2963823, upper bound: 39.2542010
time: 58.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -63.5039940, 8.5745611, -63.3593369, 8.5091400, -72.0131378, 71.9338989
1: -44.6477242, 6.6607246, -44.5314026, 6.5782843, -48.2702217, 48.2652435
2: -35.4188690, 10.6935339, -35.3443680, 10.6273422, -43.3783722, 43.4232597
3: -46.1014862, 7.4557896, -46.0768051, 7.4191656, -49.9577370, 50.0255928
4: -36.2052422, 19.6276474, -36.1022644, 19.5242596, -54.4031525, 54.4212952
5: -49.2604828, 11.0968151, -49.2097549, 11.0402594, -54.7324791, 54.7612991
6: -43.5227623, 19.1006241, -43.4785271, 18.9884472, -62.5112076, 62.5791512
7: -66.9474030, -0.3476219, -66.8634796, -0.4143353, -55.6154709, 55.6732254
8: -42.9055939, 25.0095253, -42.7682610, 24.9344711, -67.8400650, 67.7777863
9: -21.6197033, 16.4802780, -21.5349007, 16.4228878, -38.0425911, 38.0151787
10: -53.2083282, 17.4551468, -53.0110359, 17.3342819, -68.3502350, 68.3084564
11: -69.6947098, -11.5797977, -69.6161575, -11.6381626, -46.6688766, 46.6842308
12: -32.5609894, 30.1033936, -32.4982681, 30.0622177, -59.7051086, 59.6858330
13: -35.8333206, 37.3629456, -35.7494431, 37.2141342, -73.0474548, 73.1123886
14: -105.5006180, -10.9852676, -105.2827454, -11.0868759, -89.0234528, 89.0410080
15: -35.2698784, 21.8965492, -35.1442223, 21.8259735, -56.6829834, 56.6574974
16: -60.9909286, 2.2982311, -60.8535423, 2.2275019, -54.7969971, 54.7457275
17: -122.9691620, -17.6467667, -122.7199402, -17.8005123, -82.2346420, 82.1667480
18: -46.9794502, 24.2369919, -46.8637619, 24.1808815, -71.1603317, 71.1007538
19: -40.2051315, 1.7365556, -40.1397705, 1.6881456, -39.6111794, 39.5816536
20: -31.6862812, 5.3819113, -31.6421108, 5.3399153, -37.0261955, 37.0240211
21: -53.2075005, 0.2057486, -53.1215630, 0.1382446, -49.7531013, 49.7397423
22: -53.9213600, 6.1500044, -53.8323784, 6.1039095, -55.7711334, 55.6992645
23: -32.8232613, 8.2637959, -32.7709732, 8.2273598, -38.7171478, 38.6590118
24: -26.0211258, 18.4942093, -25.9825230, 18.4545479, -43.8917160, 43.8890877
25: -23.4198475, 19.7846413, -23.3767624, 19.7548294, -41.6796761, 41.6178780
26: -44.9511414, 25.2931747, -44.8453979, 25.2438622, -70.1457596, 70.0929565
27: -45.6942520, 10.9446774, -45.6485481, 10.8920898, -56.5863419, 56.5932236
28: -36.1809540, 14.3726530, -36.1280785, 14.3303328, -48.7973938, 48.7380447
29: -65.2663269, -6.1394119, -65.1631165, -6.2106266, -50.0844460, 50.0746536
30: -43.7526093, 14.3803825, -43.6394005, 14.2885036, -58.0411148, 58.0197830
31: -41.5972443, 2.8769007, -41.5444221, 2.8056865, -44.4029312, 44.4213219
32: -38.7475052, 22.5236950, -38.6899948, 22.3971367, -61.1446419, 61.2136917
33: -19.6832886, 60.1202393, -19.5063705, 59.9263687, -73.3023605, 73.2306824
34: -28.2426987, 47.5592308, -28.0616646, 47.3669243, -70.2237701, 70.2023392
35: -18.5722160, 56.0571213, -18.4091148, 55.8505020, -71.2615662, 71.2955627
36: -27.3119965, 48.2964020, -27.1906319, 48.0876541, -73.4566650, 73.5123062
37: -14.7592297, 48.4777336, -14.6032887, 48.3984756, -54.7987633, 54.6414566
38: -33.2941208, 57.6729317, -33.1605110, 57.4599075, -87.2211761, 87.3267212
39: -19.6398964, 65.8720551, -19.5213890, 65.6510773, -77.0576172, 77.1275864
40: -22.8524551, 42.2926598, -22.7283592, 42.1558533, -61.5509567, 61.5600052
41: -26.0807896, 26.2743340, -26.0243340, 26.1918907, -52.2726822, 52.2986679
42: -35.6533127, 19.5456886, -35.6027298, 19.4912720, -55.1445847, 55.1484184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=356, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1432

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2310433, upper bound: 39.2474074
time: 61.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2334203, upper bound: 39.2812339
time: 65.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 129.07 seconds
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 129.07
Output dim: 37, lower bound: -39.2167188, upper bound: 39.2204211
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 129.07
Output dim: 37, lower bound: -39.2963823, upper bound: 39.2542010
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 129.07
Output dim: 37, lower bound: -39.2310433, upper bound: 39.2474074
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 129.07
Output dim: 37, lower bound: -39.2334203, upper bound: 39.2812339
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2683065, upper bound: 39.2817651
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2081527, upper bound: 39.2850114
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2339518, upper bound: 39.2969124
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2425110, upper bound: 39.2850115
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2683065, upper bound: 39.2969124
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2625730, upper bound: 39.2817651
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2817651
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2196196, upper bound: 39.2850115
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2625730, upper bound: 39.2969124
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2539583, upper bound: 39.2850115
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.07
Output dim: 37, lower bound: -39.2969123, upper bound: 39.2969124

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 100.15 + 3559.04 = 3659.19 seconds
