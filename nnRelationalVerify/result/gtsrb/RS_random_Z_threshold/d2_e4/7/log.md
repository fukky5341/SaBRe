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
execution time: IAR + RelationalAnalysis = 2.41 + 98.71 = 101.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 37, lower bound: -39.3138009, upper bound: 39.3138009

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1714

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 807

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3127680, upper bound: 39.3136952
time: 46.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3136952, upper bound: 39.3127680
time: 81.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 127.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 127.65
Output dim: 37, lower bound: -39.3127680, upper bound: 39.3136952
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 127.65
Output dim: 37, lower bound: -39.3136952, upper bound: 39.3127680

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5146675, 48.5146561
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6728592, 43.6727295
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2232208, 50.2219543
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7369308, 54.7352371
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0847855, 55.0828476
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9495010, 55.9475708
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7369003, 68.7368469
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9721527, 46.9716339
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9854660, 59.9866333
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5045090, 89.5074997
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9754181, 56.9760208
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0708427, 55.0697899
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6152344, 82.6148987
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6664429, 39.6663857
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9258041, 49.9251404
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8689537, 55.8688202
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7860260, 38.7877274
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1038780, 44.1049881
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7578583, 41.7582436
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9114876, 48.9128761
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3049622, 50.3046379
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7459869, 73.7461090
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6918030, 70.6923599
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6728821, 71.6733780
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8693619, 73.8694916
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0143356, 55.0144577
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6900635, 87.6905289
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5185242, 77.5187988
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9517097, 61.9510918
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 721

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1170

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3078897, upper bound: 39.3135834
time: 65.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3126562, upper bound: 39.3088168
time: 66.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5146523, 48.5146713
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6727295, 43.6728592
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2219543, 50.2232132
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7352371, 54.7369308
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0828476, 55.0847893
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9475708, 55.9495010
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7368469, 68.7369003
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9716339, 46.9721489
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9866333, 59.9854660
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5074997, 89.5045090
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9760284, 56.9754143
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0697899, 55.0708351
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6148987, 82.6152344
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6663895, 39.6664429
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9251328, 49.9258118
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8688164, 55.8689461
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7877274, 38.7860298
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1049843, 44.1038780
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7582397, 41.7578545
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9128761, 48.9114876
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3046341, 50.3049660
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7461166, 73.7459869
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6923676, 70.6918030
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6733704, 71.6728821
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8694839, 73.8693695
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0144577, 55.0143280
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6905212, 87.6900711
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5187988, 77.5185242
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9510918, 61.9517097
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1714

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3132784, upper bound: 39.3008019
time: 54.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3017369, upper bound: 39.3123484
time: 68.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 125.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 125.20
Output dim: 37, lower bound: -39.3078897, upper bound: 39.3135834
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 125.20
Output dim: 37, lower bound: -39.3126562, upper bound: 39.3088168
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 125.20
Output dim: 37, lower bound: -39.3132784, upper bound: 39.3008019
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 125.20
Output dim: 37, lower bound: -39.3017369, upper bound: 39.3123484

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5087891, 48.5079422
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6705475, 43.6703339
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2228546, 50.2215614
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7348633, 54.7329712
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0891228, 55.0877342
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9458389, 55.9435349
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7269669, 68.7255096
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9799805, 46.9772758
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9824104, 59.9817619
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5039673, 89.5070038
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9765472, 56.9772110
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0655975, 55.0638618
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6146851, 82.6111450
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6669312, 39.6667938
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9257126, 49.9246864
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8714180, 55.8709564
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7854118, 38.7871513
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1048126, 44.1058693
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7558441, 41.7567444
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9054565, 48.9075394
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3121719, 50.3106041
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7426453, 73.7433929
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6884308, 70.6895981
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6649780, 71.6663742
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8644180, 73.8651352
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0105591, 55.0111694
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6856003, 87.6867447
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5171661, 77.5176086
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9545174, 61.9544373
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1649

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3015396, upper bound: 39.3124091
time: 71.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3066823, upper bound: 39.3070332
time: 46.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5079651, 48.5087662
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6704636, 43.6704178
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2228241, 50.2215996
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7346649, 54.7331772
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0896721, 55.0871849
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9454651, 55.9439087
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7255630, 68.7269135
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9777908, 46.9794655
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9805946, 59.9835777
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5040283, 89.5069504
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9766083, 56.9771500
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0649109, 55.0645409
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6114807, 82.6143417
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6668549, 39.6668701
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9253616, 49.9250450
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8710823, 55.8712921
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7854576, 38.7871056
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1047668, 44.1059227
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7563477, 41.7562408
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9061508, 48.9068489
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3109360, 50.3118477
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7432709, 73.7427673
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6890411, 70.6889801
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6658783, 71.6654663
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8650131, 73.8645325
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0110397, 55.0106850
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6862869, 87.6860504
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5173187, 77.5174408
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9550591, 61.9538994
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3019584, upper bound: 39.2849620
time: 71.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2888188, upper bound: 39.2980591
time: 63.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5242462, 48.5228844
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6724854, 43.6726151
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2218590, 50.2231216
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7411957, 54.7425842
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0754242, 55.0775528
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9510727, 55.9526443
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7310638, 68.7291412
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9501114, 46.9482574
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9650421, 59.9611626
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4996567, 89.4985657
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9777679, 56.9769745
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0661469, 55.0661430
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6268311, 82.6223145
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6662827, 39.6663055
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9203186, 49.9202919
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8588638, 55.8573227
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7913666, 38.7899246
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1038589, 44.1027298
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7617264, 41.7614746
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9187202, 48.9186630
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3029594, 50.3015060
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7480164, 73.7479553
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6978455, 70.6984177
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6702423, 71.6712189
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8697357, 73.8703003
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0154190, 55.0152435
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6877060, 87.6885681
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5174026, 77.5171356
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9577408, 61.9593086
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3127510, upper bound: 39.3006864
time: 74.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3131624, upper bound: 39.3002771
time: 68.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5228729, 48.5242615
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6724930, 43.6726112
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2218590, 50.2231216
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7408905, 54.7428894
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0756073, 55.0773621
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9507141, 55.9530029
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7290955, 68.7311249
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9477463, 46.9506264
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9623337, 59.9638748
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5015488, 89.4966583
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9775848, 56.9771538
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0651016, 55.0671921
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6219788, 82.6271667
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6662521, 39.6663399
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9196167, 49.9209900
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8572006, 55.8589935
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7916183, 38.7896652
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1038437, 44.1027489
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7618637, 41.7613373
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9200478, 48.9173355
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3011818, 50.3032799
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7480927, 73.7478790
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6989746, 70.6972961
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6717072, 71.6697540
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8704224, 73.8696213
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0153732, 55.0152817
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6890182, 87.6872635
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5174179, 77.5171204
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9586868, 61.9583588
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 743

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 741

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2978909, upper bound: 39.3094080
time: 89.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2988276, upper bound: 39.3094080
time: 63.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 155.48 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.3015396, upper bound: 39.3124091
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.3066823, upper bound: 39.3070332
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.3019584, upper bound: 39.2849620
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.2888188, upper bound: 39.2980591
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.3127510, upper bound: 39.3006864
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.3131624, upper bound: 39.3002771
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.2978909, upper bound: 39.3094080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.48
Output dim: 37, lower bound: -39.2988276, upper bound: 39.3094080

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5010986, 48.4990158
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6671524, 43.6665611
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2219124, 50.2205276
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7351494, 54.7332268
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0893860, 55.0879974
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9412155, 55.9382248
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7186508, 68.7160187
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9826736, 46.9778709
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9869843, 59.9852562
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5029602, 89.5063553
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9729156, 56.9737663
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0576706, 55.0546455
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6080322, 82.6002197
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6686974, 39.6681824
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9281883, 49.9262924
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8775406, 55.8767433
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7903366, 38.7920761
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1062927, 44.1073723
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7565231, 41.7574539
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9066353, 48.9087715
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3139038, 50.3099213
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7398987, 73.7414169
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6850128, 70.6874008
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6590500, 71.6614380
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8576050, 73.8592072
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0069580, 55.0080681
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6756058, 87.6781540
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5101166, 77.5110931
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9603386, 61.9610329
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1509

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2490227, upper bound: 39.3121005
time: 63.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3012309, upper bound: 39.2598848
time: 53.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.4998550, 48.5002594
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6667786, 43.6669426
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2218208, 50.2206116
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7351265, 54.7332535
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0893860, 55.0879974
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9405289, 55.9389114
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7174759, 68.7171936
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9805756, 46.9799652
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9859009, 59.9863358
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5033112, 89.5060043
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9730988, 56.9735832
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0563736, 55.0559425
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6037445, 82.6044922
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6683235, 39.6685600
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9273262, 49.9271584
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8772049, 55.8770866
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7903290, 38.7920837
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1063080, 44.1073341
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7565689, 41.7574120
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9066887, 48.9087181
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3114929, 50.3123283
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7406616, 73.7406616
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6862335, 70.6861801
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6600266, 71.6604538
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8584900, 73.8583221
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0074539, 55.0075760
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6770096, 87.6767578
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5106506, 77.5105438
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9611168, 61.9602547
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1601

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3065200, upper bound: 39.3023114
time: 61.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3019673, upper bound: 39.3068711
time: 70.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5077591, 48.5086708
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6704063, 43.6703987
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2222977, 50.2214546
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7345009, 54.7329483
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0896149, 55.0867119
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9449310, 55.9437599
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7233353, 68.7249603
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9757996, 46.9783783
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9801674, 59.9840164
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4985962, 89.5016632
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9776459, 56.9767532
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0642967, 55.0643158
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6110382, 82.6139450
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6654320, 39.6653900
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9238510, 49.9237671
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8650475, 55.8641510
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7852554, 38.7868881
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1045532, 44.1057587
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7554054, 41.7538071
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9056091, 48.9062767
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3080635, 50.3089371
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7406921, 73.7397003
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6885986, 70.6883698
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6657715, 71.6653290
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8649979, 73.8645172
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0095139, 55.0087738
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6855545, 87.6852493
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5172424, 77.5173798
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9546394, 61.9538155
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3012218, upper bound: 39.2561959
time: 76.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2733649, upper bound: 39.2842246
time: 71.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5078659, 48.5085640
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6704445, 43.6703606
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2226791, 50.2210732
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7344398, 54.7330055
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0892029, 55.0871315
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9453201, 55.9433746
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7236099, 68.7246857
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9767075, 46.9774666
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9810371, 59.9831543
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4987488, 89.5015106
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9762115, 56.9781914
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0646858, 55.0639305
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6110840, 82.6139069
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6653709, 39.6654510
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9240799, 49.9235306
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8639412, 55.8652534
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7852325, 38.7869072
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1045990, 44.1057129
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7539101, 41.7553024
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9055786, 48.9063072
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3080177, 50.3089752
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7401962, 73.7401962
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6884155, 70.6885376
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6657410, 71.6653519
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8649979, 73.8645172
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0091324, 55.0091591
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6854935, 87.6853333
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5172729, 77.5173492
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9549828, 61.9534798
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2848693, upper bound: 39.2940862
time: 60.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2848678, upper bound: 39.2940888
time: 50.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5269852, 48.5251083
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6664543, 43.6664238
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2073975, 50.2092285
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7251015, 54.7280464
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0541611, 55.0573044
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9368935, 55.9389191
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7274704, 68.7247467
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9418640, 46.9390030
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9502792, 59.9452324
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4888763, 89.4862823
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9774666, 56.9766617
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0651703, 55.0649147
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6245956, 82.6198120
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6672058, 39.6671410
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9229088, 49.9227371
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8594437, 55.8578300
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7815094, 38.7786751
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0968361, 44.0948563
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7611084, 41.7608032
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9105797, 48.9094315
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.2945099, 50.2919350
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7451935, 73.7454300
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6957779, 70.6965485
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6709137, 71.6719666
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8681641, 73.8690796
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0180435, 55.0180588
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6890945, 87.6901627
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5059814, 77.5071487
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9606628, 61.9628983
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3083640, upper bound: 39.2988443
time: 56.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3109088, upper bound: 39.2964386
time: 51.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5264664, 48.5256271
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6662941, 43.6665840
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2079620, 50.2086563
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7266579, 54.7264862
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0551682, 55.0563011
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9373512, 55.9384651
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7266617, 68.7255402
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9408569, 46.9400024
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9491043, 59.9464035
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4873810, 89.4877930
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9774513, 56.9766884
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0649185, 55.0651665
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6243210, 82.6200867
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6671143, 39.6672401
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9227562, 49.9228859
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8593674, 55.8578987
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7801170, 38.7800636
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0959816, 44.0957146
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7610626, 41.7608490
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9094963, 48.9105148
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.2933807, 50.2930603
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7454987, 73.7451324
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6959763, 70.6963501
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6710052, 71.6718750
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8685303, 73.8687210
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0182266, 55.0178871
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6893082, 87.6899490
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5074005, 77.5057144
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9613342, 61.9622307
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1743

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 686

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3103648, upper bound: 39.2979426
time: 67.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3108152, upper bound: 39.2974765
time: 55.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5224762, 48.5239296
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6712265, 43.6714897
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2212448, 50.2225113
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7399597, 54.7429543
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0749588, 55.0767822
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9497604, 55.9522362
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7282028, 68.7303085
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9459190, 46.9490356
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9621277, 59.9636841
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4964447, 89.4915619
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9769592, 56.9772568
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0631409, 55.0654068
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6181488, 82.6235962
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6657028, 39.6659126
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9191246, 49.9205589
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8564301, 55.8588409
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7902985, 38.7885361
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1038055, 44.1025620
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7612457, 41.7613220
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9189262, 48.9164085
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3007736, 50.3030014
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7463074, 73.7458344
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6983185, 70.6965561
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6712341, 71.6691895
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8695831, 73.8686981
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0128403, 55.0127449
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6883163, 87.6864777
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5148849, 77.5143280
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9573212, 61.9567986
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1449

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2913604, upper bound: 39.3069102
time: 53.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2963345, upper bound: 39.3028710
time: 57.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5225372, 48.5242615
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6713715, 43.6726112
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2212448, 50.2231216
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7408905, 54.7419701
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0750351, 55.0773621
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9499512, 55.9530029
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7282791, 68.7311249
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9461555, 46.9506264
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9623337, 59.9636688
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4964600, 89.4966583
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9775848, 56.9765358
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0633163, 55.0671921
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6184082, 82.6271667
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6658249, 39.6663399
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9191856, 49.9209900
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8572006, 55.8582268
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7904892, 38.7896652
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1036530, 44.1027489
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7618637, 41.7607269
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9191170, 48.9173355
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3008957, 50.3032799
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7480927, 73.7460938
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6989746, 70.6966400
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6717072, 71.6692734
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8704224, 73.8687897
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0153732, 55.0127525
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6890182, 87.6865692
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5174179, 77.5145874
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9586868, 61.9569931
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2976354, upper bound: 39.3078994
time: 62.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2975816, upper bound: 39.3091810
time: 69.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 134.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2490227, upper bound: 39.3121005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3012309, upper bound: 39.2598848
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3065200, upper bound: 39.3023114
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3019673, upper bound: 39.3068711
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3012218, upper bound: 39.2561959
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2733649, upper bound: 39.2842246
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2848693, upper bound: 39.2940862
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2848678, upper bound: 39.2940888
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3083640, upper bound: 39.2988443
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3109088, upper bound: 39.2964386
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3103648, upper bound: 39.2979426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.3108152, upper bound: 39.2974765
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2913604, upper bound: 39.3069102
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2963345, upper bound: 39.3028710
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2976354, upper bound: 39.3078994
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 134.30
Output dim: 37, lower bound: -39.2975816, upper bound: 39.3091810

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5043526, 48.5024300
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6741486, 43.6697617
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2072449, 50.2019501
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7122726, 54.7081223
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1094513, 55.1030960
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9473724, 55.9429016
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7185516, 68.7155991
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9706268, 46.9631767
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9726639, 59.9748039
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5170898, 89.5144806
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9629135, 56.9650383
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0588112, 55.0554314
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.5728760, 82.5595627
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6667709, 39.6646233
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9261780, 49.9241295
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8687630, 55.8687515
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7901154, 38.7917824
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1053238, 44.1063576
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7558556, 41.7569580
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9081573, 48.9098587
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3141899, 50.3101196
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7396774, 73.7412415
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6909637, 70.6942749
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6580200, 71.6605606
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8597031, 73.8617249
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -54.9665146, 54.9724922
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6863098, 87.6904831
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5075150, 77.5100937
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9507675, 61.9529305
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 808

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2428979, upper bound: 39.3118410
time: 78.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2487632, upper bound: 39.3052245
time: 48.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5045128, 48.5022621
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6703491, 43.6735535
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2033386, 50.2058601
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7100449, 54.7103386
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1044922, 55.1080589
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9458923, 55.9443741
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7182312, 68.7159119
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9679718, 46.9658279
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9765244, 59.9709435
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5110779, 89.5204926
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9641800, 56.9637718
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0584602, 55.0557823
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.5673828, 82.5650635
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6651382, 39.6662598
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9260254, 49.9242783
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8695564, 55.8679581
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7900467, 38.7918510
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1052780, 44.1064034
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7560234, 41.7567825
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9077301, 48.9102859
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3140984, 50.3102150
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7397232, 73.7411957
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6918945, 70.6933594
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6581726, 71.6604004
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8601303, 73.8613205
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -54.9713821, 54.9676247
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6879578, 87.6888428
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5091019, 77.5085068
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9522324, 61.9514732
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1095

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1633

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3000256, upper bound: 39.2594481
time: 64.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3007885, upper bound: 39.2587019
time: 55.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5029106, 48.5034180
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6647606, 43.6650200
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2271957, 50.2264442
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7208176, 54.7191353
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0785522, 55.0783424
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9388504, 55.9372787
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7227631, 68.7225342
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -47.0029564, 47.0026436
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9770699, 59.9768715
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5192719, 89.5216064
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9612427, 56.9630394
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0561981, 55.0557747
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.5981369, 82.5993500
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6659164, 39.6663742
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9258614, 49.9257774
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8627396, 55.8645134
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7958870, 38.7979012
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1115265, 44.1122932
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7477837, 41.7497215
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9087410, 48.9110184
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3115387, 50.3123741
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7519608, 73.7518616
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6826019, 70.6822815
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6627502, 71.6631317
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8609924, 73.8607483
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0095444, 55.0097313
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6714096, 87.6707077
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5229263, 77.5223846
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9569397, 61.9555511
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1515

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3048030, upper bound: 39.3005666
time: 57.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3047796, upper bound: 39.3005889
time: 57.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5030174, 48.5033188
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6648521, 43.6649284
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2276688, 50.2259750
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7210083, 54.7189484
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0797272, 55.0771637
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9388962, 55.9372368
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7228165, 68.7224808
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -47.0032616, 47.0023499
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9764442, 59.9775009
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.5189056, 89.5219574
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9625549, 56.9617271
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0562134, 55.0557671
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.5986099, 82.5988770
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6661301, 39.6661530
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9259453, 49.9256935
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8646240, 55.8626251
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7961502, 38.7976341
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1112747, 44.1125488
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7488823, 41.7486305
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9089928, 48.9107590
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3115387, 50.3123665
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7518387, 73.7519760
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6823273, 70.6825562
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6627045, 71.6631775
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8609161, 73.8608246
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0096054, 55.0096703
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6709518, 87.6711502
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5224991, 77.5228195
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9564056, 61.9560814
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1263

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1369

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3011845, upper bound: 39.3067848
time: 66.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3018810, upper bound: 39.3060885
time: 73.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5132332, 48.5141296
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6769257, 43.6767044
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2412262, 50.2393951
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7642288, 54.7595558
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1131630, 55.1089478
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9679031, 55.9660110
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7270508, 68.7286377
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9757538, 46.9784393
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9812317, 59.9851418
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4599915, 89.4668808
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9813995, 56.9801254
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0805016, 55.0802994
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6397095, 82.6419373
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6659393, 39.6659126
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9279099, 49.9276390
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8650894, 55.8638535
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7849770, 38.7887955
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1143265, 44.1164513
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7565994, 41.7551498
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9213181, 48.9246674
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3096123, 50.3105125
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7380447, 73.7369690
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7028275, 70.7029419
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6768188, 71.6765747
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8660736, 73.8655853
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0159073, 55.0151482
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7015839, 87.7016602
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5015717, 77.5001144
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9548454, 61.9540291
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 574

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2576262, upper bound: 39.2558685
time: 58.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3008943, upper bound: 39.2125904
time: 68.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5132179, 48.5141487
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6767120, 43.6769104
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2402344, 50.2403908
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7611008, 54.7626839
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.1118507, 55.1102524
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9671860, 55.9667358
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7270126, 68.7286682
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9758606, 46.9783363
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9812927, 59.9850807
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4637909, 89.4630814
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9810181, 56.9805031
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0802879, 55.0805168
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6390381, 82.6426010
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6659622, 39.6658974
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9277191, 49.9278336
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8647461, 55.8641930
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7871666, 38.7866058
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.1152496, 44.1155319
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7567673, 41.7549896
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9240036, 48.9219894
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3096275, 50.3104935
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7379532, 73.7370605
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7031631, 70.7026062
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6770172, 71.6763763
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8660583, 73.8656006
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0158920, 55.0151558
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7019806, 87.7012787
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4999847, 77.5017014
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9548454, 61.9540291
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2558568, upper bound: 39.2838398
time: 97.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -39.2729803, upper bound: 39.2667181
time: 55.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5227051, 48.5268707
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6779823, 43.6795082
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2212753, 50.2196732
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7210045, 54.7188492
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0826874, 55.0802002
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9521866, 55.9509354
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7170258, 68.7219238
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9284477, 46.9356308
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9604645, 59.9659386
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4454651, 89.4526901
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9670448, 56.9698677
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0705795, 55.0730019
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.4999390, 82.5160065
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6393356, 39.6420364
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9029045, 49.9044418
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8302879, 55.8353615
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7475471, 38.7522774
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0927925, 44.0952759
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7506790, 41.7521133
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.8802872, 48.8812943
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.2395439, 50.2484207
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7446136, 73.7424622
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.7020416, 70.6988144
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6739578, 71.6709290
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8599243, 73.8564606
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0033188, 55.0009689
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6580200, 87.6539230
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4798279, 77.4749756
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9131126, 61.9050865
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1618

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2696651, upper bound: 39.2869596
time: 61.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2776659, upper bound: 39.2789335
time: 73.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5261688, 48.5234032
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6795921, 43.6778984
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2212830, 50.2196693
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7202797, 54.7195740
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0822601, 55.0806198
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9528732, 55.9502487
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7208405, 68.7181015
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9348717, 46.9292107
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9638214, 59.9625931
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4499054, 89.4482498
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9678841, 56.9690285
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0737534, 55.0698242
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.5131836, 82.5027695
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6419525, 39.6394196
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9049950, 49.9023628
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8340492, 55.8315964
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7506065, 38.7492180
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0941582, 44.0939064
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7507324, 41.7520523
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.8805695, 48.8810081
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.2474632, 50.2404976
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7424622, 73.7446136
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6987000, 70.7021484
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6713181, 71.6735687
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8569336, 73.8594513
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0009308, 55.0033493
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6540833, 87.6578674
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.4748840, 77.4799042
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9065895, 61.9116135
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1448

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2696634, upper bound: 39.2869623
time: 69.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2776643, upper bound: 39.2789362
time: 60.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5051880, 48.5033836
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6647530, 43.6647415
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2029572, 50.2043037
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7276268, 54.7299805
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0557365, 55.0586815
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9024658, 55.9003944
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7183990, 68.7162476
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9694824, 46.9645119
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9394836, 59.9356613
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4779358, 89.4767151
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9547653, 56.9567719
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0535240, 55.0527573
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6447296, 82.6410141
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6717072, 39.6718330
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9420509, 49.9417763
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8495903, 55.8489838
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7835197, 38.7809067
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0932617, 44.0917282
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7448502, 41.7465668
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9116516, 48.9106216
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3073997, 50.3055801
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7639771, 73.7643814
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6989746, 70.6995316
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6717758, 71.6734009
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8767090, 73.8773346
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0130424, 55.0130730
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7021637, 87.7029953
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5343933, 77.5351639
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9603271, 61.9624290
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1727

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3083093, upper bound: 39.2873045
time: 52.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.2968290, upper bound: 39.2987896
time: 68.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5052643, 48.5033035
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6647758, 43.6647224
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2024689, 50.2047997
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7270317, 54.7305717
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0555382, 55.0588799
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.8983688, 55.9044914
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7189789, 68.7156754
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9673615, 46.9666328
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9407043, 59.9344406
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4793091, 89.4753418
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9575882, 56.9539566
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0530128, 55.0532722
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6457977, 82.6399384
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6719055, 39.6716347
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9419518, 49.9418793
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8505974, 55.8479805
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7837372, 38.7806892
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0937042, 44.0912857
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7468643, 41.7445450
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9117661, 48.9105072
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.3081474, 50.3048286
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7641373, 73.7642212
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6987610, 70.6997375
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6723404, 71.6728210
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8764038, 73.8776398
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0130577, 55.0130501
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.7019196, 87.7032547
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5339966, 77.5355759
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9601898, 61.9625549
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3020737, upper bound: 39.2963313
time: 59.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3108015, upper bound: 39.2876026
time: 47.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5262604, 48.5254211
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6636810, 43.6638031
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2065773, 50.2070160
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7271042, 54.7269783
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0524864, 55.0531349
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9343872, 55.9351120
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7258301, 68.7249146
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9435730, 46.9427032
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9436417, 59.9416008
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4859390, 89.4862671
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9775467, 56.9769516
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0673370, 55.0672798
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6243134, 82.6201019
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6681824, 39.6681252
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9245148, 49.9244690
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8571243, 55.8558960
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7818108, 38.7817078
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0955391, 44.0953560
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7608795, 41.7606201
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9105835, 48.9113579
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.2935143, 50.2932320
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7478790, 73.7477570
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6949387, 70.6954498
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6712494, 71.6721344
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8681030, 73.8682785
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0168915, 55.0166359
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6880035, 87.6887589
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5114288, 77.5101318
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9610291, 61.9619026
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1516

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3084825, upper bound: 39.2955951
time: 129.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3080096, upper bound: 39.2960691
time: 68.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.6144943, 8.6127901, -63.6144943, 8.6127901, -72.2272873, 72.2272873
1: -44.7331390, 6.6902580, -44.7331390, 6.6902580, -48.5262604, 48.5254173
2: -35.4903641, 10.7305183, -35.4903641, 10.7305183, -43.6635132, 43.6639748
3: -46.1417389, 7.4998426, -46.1417389, 7.4998426, -50.2063255, 50.2072678
4: -36.3329926, 19.6753159, -36.3329926, 19.6753159, -54.7271576, 54.7269287
5: -49.3284149, 11.1488447, -49.3284149, 11.1488447, -55.0519981, 55.0536156
6: -43.5636673, 19.2334080, -43.5636673, 19.2334080, -62.7970734, 62.7970734
7: -67.0328522, -0.3063717, -67.0328522, -0.3063717, -55.9339905, 55.9355011
8: -43.0104866, 25.0567741, -43.0104866, 25.0567741, -68.0672607, 68.0672607
9: -21.6874542, 16.5061283, -21.6874542, 16.5061283, -38.1935806, 38.1935806
10: -53.3798866, 17.5079823, -53.3798866, 17.5079823, -68.7260361, 68.7247162
11: -69.7829666, -11.5452700, -69.7829666, -11.5452700, -46.9435577, 46.9427223
12: -32.6403275, 30.1528683, -32.6403275, 30.1528683, -59.9443130, 59.9409294
13: -35.8913193, 37.4639549, -35.8913193, 37.4639549, -73.3552704, 73.3552704
14: -105.7178650, -10.9548473, -105.7178650, -10.9548473, -89.4858475, 89.4863586
15: -35.3873024, 21.9407921, -35.3873024, 21.9407921, -56.9777069, 56.9767952
16: -61.1410522, 2.3462658, -61.1410522, 2.3462658, -55.0670319, 55.0675888
17: -123.2366180, -17.5973301, -123.2366180, -17.5973301, -82.6243286, 82.6200867
18: -47.1038208, 24.2779636, -47.1038208, 24.2779636, -71.3817825, 71.3817825
19: -40.2941170, 1.7601848, -40.2941170, 1.7601848, -39.6679993, 39.6683044
20: -31.7336197, 5.4156637, -31.7336197, 5.4156637, -37.1492844, 37.1492844
21: -53.3338203, 0.2291021, -53.3338203, 0.2291021, -49.9243469, 49.9246445
22: -54.1081772, 6.1827221, -54.1081772, 6.1827221, -55.8573685, 55.8556404
23: -32.8686905, 8.2988911, -32.8686905, 8.2988911, -38.7817574, 38.7817612
24: -26.1320229, 18.5359726, -26.1320229, 18.5359726, -44.0956230, 44.0952682
25: -23.4801750, 19.8237457, -23.4801750, 19.8237457, -41.7608337, 41.7606659
26: -45.0802155, 25.3260899, -45.0802155, 25.3260899, -70.4063034, 70.4063034
27: -45.8041496, 10.9876537, -45.8041496, 10.9876537, -56.7918015, 56.7918015
28: -36.2330360, 14.4241438, -36.2330360, 14.4241438, -48.9103317, 48.9116096
29: -65.4423370, -6.1152134, -65.4423370, -6.1152134, -50.2935524, 50.2931976
30: -43.8592606, 14.4851112, -43.8592606, 14.4851112, -58.3443718, 58.3443718
31: -41.6757584, 2.9048014, -41.6757584, 2.9048014, -44.5805588, 44.5805588
32: -38.8264503, 22.6317062, -38.8264503, 22.6317062, -61.4581566, 61.4581566
33: -19.7550621, 60.2852173, -19.7550621, 60.2852173, -73.7481155, 73.7475204
34: -28.3038406, 47.7331657, -28.3038406, 47.7331657, -70.6950912, 70.6953049
35: -18.6240730, 56.2216415, -18.6240730, 56.2216415, -71.6712494, 71.6721268
36: -27.3568954, 48.4599533, -27.3568954, 48.4599533, -73.8680878, 73.8683014
37: -14.8436012, 48.5743523, -14.8436012, 48.5743523, -55.0169678, 55.0165520
38: -33.3484192, 57.8838120, -33.3484192, 57.8838120, -87.6881256, 87.6886520
39: -19.7234783, 66.0237122, -19.7234783, 66.0237122, -77.5118103, 77.5097504
40: -22.9293747, 42.4303513, -22.9293747, 42.4303513, -61.9609985, 61.9619255
41: -26.1342564, 26.3741493, -26.1342564, 26.3741493, -52.5084076, 52.5084076
42: -35.7053032, 19.6478767, -35.7053032, 19.6478767, -55.3531799, 55.3531799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=38, inp2_unstable=38, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3107524, upper bound: 39.2880896
time: 61.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -39.3014297, upper bound: 39.2974136
time: 64.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 128.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2428979, upper bound: 39.3118410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2487632, upper bound: 39.3052245
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3000256, upper bound: 39.2594481
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3007885, upper bound: 39.2587019
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3048030, upper bound: 39.3005666
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3047796, upper bound: 39.3005889
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3011845, upper bound: 39.3067848
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3018810, upper bound: 39.3060885
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2576262, upper bound: 39.2558685
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3008943, upper bound: 39.2125904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2558568, upper bound: 39.2838398
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2729803, upper bound: 39.2667181
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2696651, upper bound: 39.2869596
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2776659, upper bound: 39.2789335
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2696634, upper bound: 39.2869623
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2776643, upper bound: 39.2789362
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3083093, upper bound: 39.2873045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.2968290, upper bound: 39.2987896
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3020737, upper bound: 39.2963313
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3108015, upper bound: 39.2876026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3084825, upper bound: 39.2955951
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3080096, upper bound: 39.2960691
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3107524, upper bound: 39.2880896
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 128.41
Output dim: 37, lower bound: -39.3014297, upper bound: 39.2974136
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.41
Output dim: 37, lower bound: -39.2913604, upper bound: 39.3069102
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.41
Output dim: 37, lower bound: -39.2963345, upper bound: 39.3028710
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.41
Output dim: 37, lower bound: -39.2976354, upper bound: 39.3078994
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.41
Output dim: 37, lower bound: -39.2975816, upper bound: 39.3091810

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 101.12 + 3568.49 = 3669.61 seconds
