## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 20.1836426535


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498)
1: (-11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070)
2: (-9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425)
3: (-9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4056396, 38.4056396)
4: (-16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9559555, 41.9559555)
5: (-7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1119690, 36.1119690)
6: (-38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812)
7: (-11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.6014786, 38.6014748)
8: (-21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7545319, 50.7545319)
9: (-13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186)
10: (-22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662)
11: (-23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790)
12: (-44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2175751, 45.2175751)
13: (-37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5576782, 59.5576782)
14: (-64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507)
15: (-21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186)
16: (-23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021)
17: (-58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0626373, 56.0626373)
18: (-35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650)
19: (-26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775)
20: (-21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115)
21: (-27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035)
22: (-32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925)
23: (-24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843)
24: (-30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705)
25: (-28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095)
26: (-41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693)
27: (-26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029)
28: (-25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052)
29: (-27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4229507, 38.4229507)
30: (-26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750)
31: (-35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546)
32: (-35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8387909, 45.8387985)
33: (-63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3612823, 55.3612900)
34: (-57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5909424, 47.5909424)
35: (-56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8676910, 44.8676949)
36: (-53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4313049, 49.4312973)
37: (-78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8089600, 60.8089523)
38: (-63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6764984, 59.6764984)
39: (-72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0168457, 58.0168381)
40: (-51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460)
41: (-40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611)
42: (-26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.63 + 56.21 = 58.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -20.2038465, upper bound: 20.2038465

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1549411, upper bound: 20.2023272
time: 50.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2023272, upper bound: 20.1549411
time: 47.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 98.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 98.23
Output dim: 5, lower bound: -20.1549411, upper bound: 20.2023272
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 98.23
Output dim: 5, lower bound: -20.2023272, upper bound: 20.1549411

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4060974, 38.4061050
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9550018, 41.9550934
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1094666, 36.1097336
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5953522, 38.5960236
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7565231, 50.7567368
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2178040, 45.2174606
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5444336, 59.5437012
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0679092, 56.0673599
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4200821, 38.4196548
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8397980, 45.8395538
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3580475, 55.3576279
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5763550, 47.5743179
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8621063, 44.8614349
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4408112, 49.4402618
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8029327, 60.8023682
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6934967, 59.6929398
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0216522, 58.0208740
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1024880, upper bound: 20.1989167
time: 56.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1515333, upper bound: 20.1498606
time: 50.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4061050, 38.4061012
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9550934, 41.9549980
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1097412, 36.1094666
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5960236, 38.5953522
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7567368, 50.7565231
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2174683, 45.2178116
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5437012, 59.5444336
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0673599, 56.0679169
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4196548, 38.4200821
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8395538, 45.8398056
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3576202, 55.3580475
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5743103, 47.5763550
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8614349, 44.8621063
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4402618, 49.4408112
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8023682, 60.8029327
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6929474, 59.6935043
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0208740, 58.0216522
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1498606, upper bound: 20.1515333
time: 52.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1989167, upper bound: 20.1024880
time: 76.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 131.72 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 131.72
Output dim: 5, lower bound: -20.1024880, upper bound: 20.1989167
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 131.72
Output dim: 5, lower bound: -20.1515333, upper bound: 20.1498606
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 131.72
Output dim: 5, lower bound: -20.1498606, upper bound: 20.1515333
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 131.72
Output dim: 5, lower bound: -20.1989167, upper bound: 20.1024880

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4036560, 38.4050636
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9545212, 41.9547462
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1095695, 36.1096878
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5954514, 38.5958633
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7585678, 50.7565460
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2177734, 45.2176208
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5434723, 59.5457687
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0655670, 56.0640526
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4200821, 38.4196548
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8395538, 45.8417130
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3515320, 55.3559036
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5699463, 47.5713577
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8568573, 44.8585854
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4399567, 49.4403534
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8024902, 60.8019943
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6934967, 59.6929474
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0199738, 58.0238037
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0586482, upper bound: 20.1976153
time: 46.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1011706, upper bound: 20.1544479
time: 44.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4050674, 38.4036560
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9547424, 41.9545212
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1096916, 36.1095657
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5958633, 38.5954552
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7565384, 50.7585602
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2176208, 45.2177734
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5457611, 59.5434799
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0640411, 56.0655823
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4196548, 38.4200821
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8417206, 45.8395615
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3558960, 55.3515320
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5713501, 47.5699539
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8585892, 44.8568573
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4403534, 49.4399567
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8019867, 60.8024902
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6929474, 59.6934814
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0238037, 58.0199661
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1544479, upper bound: 20.1011706
time: 53.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1976153, upper bound: 20.0586482
time: 41.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 97.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 97.52
Output dim: 5, lower bound: -20.0586482, upper bound: 20.1976153
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 97.52
Output dim: 5, lower bound: -20.1011706, upper bound: 20.1544479
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 97.52
Output dim: 5, lower bound: -20.1544479, upper bound: 20.1011706
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 97.52
Output dim: 5, lower bound: -20.1976153, upper bound: 20.0586482

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.3995285, 38.4015312
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9498901, 41.9508247
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1023712, 36.1036224
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5846100, 38.5867271
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7533264, 50.7521591
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2045670, 45.2014084
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5540009, 59.5546417
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0935059, 56.0875206
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4165344, 38.4154434
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8397675, 45.8419113
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3392792, 55.3413010
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5497742, 47.5472794
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8384399, 44.8366547
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4320984, 49.4310532
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7840576, 60.7801590
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6842194, 59.6820297
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0118027, 58.0141830
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0287720, upper bound: 20.1967663
time: 68.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0576644, upper bound: 20.1483650
time: 55.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4015350, 38.3995285
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9508209, 41.9498901
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1036224, 36.1023712
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5867233, 38.5846100
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7521515, 50.7533340
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2014084, 45.2045670
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5546417, 59.5539856
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0875244, 56.0935097
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4154434, 38.4165344
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8419037, 45.8397827
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3413086, 55.3392792
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5472717, 47.5497818
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8366547, 44.8384399
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4310455, 49.4320984
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7801514, 60.7840576
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6820374, 59.6842117
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0141830, 58.0118103
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1483650, upper bound: 20.0576644
time: 67.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1967663, upper bound: 20.0287720
time: 75.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 146.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 146.06
Output dim: 5, lower bound: -20.0287720, upper bound: 20.1967663
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 146.06
Output dim: 5, lower bound: -20.0576644, upper bound: 20.1483650
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 146.06
Output dim: 5, lower bound: -20.1483650, upper bound: 20.0576644
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 146.06
Output dim: 5, lower bound: -20.1967663, upper bound: 20.0287720

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4004974, 38.4024429
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9479446, 41.9491997
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0977249, 36.0997543
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5773544, 38.5806923
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7478027, 50.7475281
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2054901, 45.2004547
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5650024, 59.5642548
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1067963, 56.0988731
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4144592, 38.4129486
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8362579, 45.8379288
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3253860, 55.3247986
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5304260, 47.5240784
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8224945, 44.8175888
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4264221, 49.4245682
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7736664, 60.7678452
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6791382, 59.6766510
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0015411, 58.0023270
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1551387
time: 44.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1953748
time: 55.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4024429, 38.4004974
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9491959, 41.9479446
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0997543, 36.0977249
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5806885, 38.5773544
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7475281, 50.7478027
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2004547, 45.2054901
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5642548, 59.5650101
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0988770, 56.1067924
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4129486, 38.4144592
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8379364, 45.8362656
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3248062, 55.3253784
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5240784, 47.5304413
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8175888, 44.8224945
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4245605, 49.4264145
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7678528, 60.7736664
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6766510, 59.6791382
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0023346, 58.0015488
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1463188, upper bound: 19.9871358
time: 50.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1551387, upper bound: 20.0273819
time: 56.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 109.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 109.31
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1551387
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 109.31
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1953748
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 109.31
Output dim: 5, lower bound: -20.1463188, upper bound: 19.9871358
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 109.31
Output dim: 5, lower bound: -20.1551387, upper bound: 20.0273819

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.3915405, 38.3949280
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9399109, 41.9424515
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0864487, 36.0902786
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5625458, 38.5682411
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7364120, 50.7378845
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1781158, 45.1678467
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5921936, 59.5869217
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0620346, 56.0459785
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4102402, 38.4079285
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8358612, 45.8375244
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3147202, 55.3107834
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5051422, 47.4941254
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8018036, 44.7914352
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4091492, 49.4029922
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7438507, 60.7322388
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6530914, 59.6445541
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9898987, 57.9869766
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9341449, upper bound: 20.1255935
time: 47.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9341449, upper bound: 20.1238062
time: 53.21 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 102.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 102.69
Output dim: 5, lower bound: -19.9341449, upper bound: 20.1255935
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 102.69
Output dim: 5, lower bound: -19.9341449, upper bound: 20.1238062

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 58.84 + 1117.02 = 1175.87 seconds
