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
execution time: IAR + RelationalAnalysis = 2.82 + 56.49 = 59.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -20.2038465, upper bound: 20.2038465

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1541

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 899

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1850643, upper bound: 20.2030132
time: 50.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2030132, upper bound: 20.1850643
time: 46.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 97.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 97.30
Output dim: 5, lower bound: -20.1850643, upper bound: 20.2030132
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 97.30
Output dim: 5, lower bound: -20.2030132, upper bound: 20.1850643

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4050369, 38.4054184
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9550171, 41.9556580
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1102753, 36.1113510
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5997849, 38.6011810
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7519608, 50.7536087
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2156906, 45.2125778
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5581207, 59.5542145
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0640869, 56.0600777
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4229050, 38.4225273
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8382034, 45.8376236
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3591766, 55.3559875
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5893250, 47.5855331
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8652725, 44.8612289
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4296112, 49.4265518
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8071594, 60.8040085
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6735382, 59.6698532
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0142975, 58.0104980
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1770578, upper bound: 20.2024946
time: 58.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1845476, upper bound: 20.1950043
time: 53.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4054184, 38.4050369
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9556580, 41.9550171
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1113510, 36.1102753
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.6011810, 38.5997887
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7536087, 50.7519531
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2125626, 45.2156906
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5542145, 59.5581207
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0600739, 56.0640869
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4225235, 38.4229050
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8376236, 45.8382034
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3559723, 55.3591843
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5855408, 47.5893250
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8612289, 44.8652725
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4265594, 49.4296265
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8040161, 60.8071518
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6698456, 59.6735535
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0105133, 58.0142975
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1602

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2024757, upper bound: 20.1848029
time: 47.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2027518, upper bound: 20.1845230
time: 49.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 99.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 99.53
Output dim: 5, lower bound: -20.1770578, upper bound: 20.2024946
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 99.53
Output dim: 5, lower bound: -20.1845476, upper bound: 20.1950043
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 99.53
Output dim: 5, lower bound: -20.2024757, upper bound: 20.1848029
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 99.53
Output dim: 5, lower bound: -20.2027518, upper bound: 20.1845230

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4025803, 38.4040565
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9542542, 41.9558182
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1072845, 36.1096039
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5981064, 38.6010170
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7516098, 50.7547302
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2091217, 45.2022552
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5500870, 59.5421143
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0629501, 56.0530930
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4227905, 38.4220123
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8371429, 45.8365631
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3531723, 55.3481369
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5899582, 47.5838776
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8629074, 44.8550758
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4241486, 49.4174881
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7998657, 60.7926331
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6665802, 59.6577377
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0065079, 57.9990997
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
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1753648, upper bound: 20.1994402
time: 52.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1740033, upper bound: 20.2008016
time: 52.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4036789, 38.4029579
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9551773, 41.9548950
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1085281, 36.1083603
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5996246, 38.5994987
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7530746, 50.7532578
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2053833, 45.2060089
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5460281, 59.5461731
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0570908, 56.0589447
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4223862, 38.4224167
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8371429, 45.8365631
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3513412, 55.3499756
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5876694, 47.5861740
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8591156, 44.8588638
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4205627, 49.4210892
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7957764, 60.7967300
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6614380, 59.6628799
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0029068, 58.0027008
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 546

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1826819, upper bound: 20.1931322
time: 54.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1751913, upper bound: 20.1931322
time: 50.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4054184, 38.4051437
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9530334, 41.9527931
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1120834, 36.1117516
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5965881, 38.5957985
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7394791, 50.7399063
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1977921, 45.1984787
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5226288, 59.5211945
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0558167, 56.0573387
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4225616, 38.4229355
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8302460, 45.8294907
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3378754, 55.3378220
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5893250, 47.5907288
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8413773, 44.8418808
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4025192, 49.4015884
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7919922, 60.7930756
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6392136, 59.6373749
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9791870, 57.9776764
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 866

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1887709, upper bound: 20.1847026
time: 49.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2023753, upper bound: 20.1710907
time: 88.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4055328, 38.4050369
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9534302, 41.9524002
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1128311, 36.1110115
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5971909, 38.5951920
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7415543, 50.7378235
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1953506, 45.2009201
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5172882, 59.5265198
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0533447, 56.0598373
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4225616, 38.4229431
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8289032, 45.8308334
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3346252, 55.3410721
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5869293, 47.5931091
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8378372, 44.8454208
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3985214, 49.4056015
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7899323, 60.7951431
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6336899, 59.6429062
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9738770, 57.9829941
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2015578, upper bound: 20.1677302
time: 49.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1859805, upper bound: 20.1833278
time: 46.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 98.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.1753648, upper bound: 20.1994402
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.1740033, upper bound: 20.2008016
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.1826819, upper bound: 20.1931322
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.1751913, upper bound: 20.1931322
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.1887709, upper bound: 20.1847026
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.2023753, upper bound: 20.1710907
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.2015578, upper bound: 20.1677302
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 98.35
Output dim: 5, lower bound: -20.1859805, upper bound: 20.1833278

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4033966, 38.4048615
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9542999, 41.9558525
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1065521, 36.1087265
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5968018, 38.5994720
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7484894, 50.7513657
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2066574, 45.2000809
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5531006, 59.5455017
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0624847, 56.0527306
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4225998, 38.4218521
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8339233, 45.8333740
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3453369, 55.3410797
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5812683, 47.5764465
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8587189, 44.8516197
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4236908, 49.4172134
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7985382, 60.7915039
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6651001, 59.6563873
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0014191, 57.9944153
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 704

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1728273, upper bound: 20.1967525
time: 53.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1726687, upper bound: 20.1968987
time: 54.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4033813, 38.4048729
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9542923, 41.9558601
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1064072, 36.1088715
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5965576, 38.5997162
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7482452, 50.7516174
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2069473, 45.1997910
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5534821, 59.5451355
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0625763, 56.0526352
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4226303, 38.4218140
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8339539, 45.8333359
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3461151, 55.3403015
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5825348, 47.5751801
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8594513, 44.8508835
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4238739, 49.4170151
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7987518, 60.7912979
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6652374, 59.6562500
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0018158, 57.9940109
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1695033, upper bound: 20.1876464
time: 52.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1608541, upper bound: 20.1962885
time: 57.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4026947, 38.4019432
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9550476, 41.9546509
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1075745, 36.1068802
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5985718, 38.5978088
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7496033, 50.7483597
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2013245, 45.2034149
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5364456, 59.5395813
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0569992, 56.0594826
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222794, 38.4223328
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8332367, 45.8336182
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3418503, 55.3432770
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5786057, 47.5800858
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8499298, 44.8523788
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4137878, 49.4162750
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7927399, 60.7947464
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6534424, 59.6571579
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9914322, 57.9944687
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1722

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1191002, upper bound: 20.1294592
time: 65.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1191002, upper bound: 20.1294592
time: 65.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4026642, 38.4019737
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9549332, 41.9547653
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1070404, 36.1074142
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5979385, 38.5984459
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7481689, 50.7497864
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2027893, 45.2019424
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5394363, 59.5365982
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0576401, 56.0588570
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4223022, 38.4223022
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8341827, 45.8326721
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3446274, 55.3404922
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5815811, 47.5771027
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8526306, 44.8496819
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4157562, 49.4143066
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7937927, 60.7936935
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6557312, 59.6548691
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9946671, 57.9912415
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 681

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 777

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1806733, upper bound: 20.1875768
time: 66.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1771301, upper bound: 20.1911133
time: 65.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4053574, 38.4050980
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9531326, 41.9529076
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1126823, 36.1123505
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5972290, 38.5963402
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7394180, 50.7398453
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1967239, 45.1973495
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5240784, 59.5225601
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0544739, 56.0558815
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4224014, 38.4228363
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8302841, 45.8295593
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3390961, 55.3391800
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5922165, 47.5940781
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8420715, 44.8426819
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4024353, 49.4013443
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7927246, 60.7937164
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6384506, 59.6362534
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9781494, 57.9765778
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1387

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 576

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1850301, upper bound: 20.1809879
time: 54.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1850301, upper bound: 20.1809879
time: 55.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4053726, 38.4050903
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9531555, 41.9528885
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1126900, 36.1123466
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5971298, 38.5964394
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7394180, 50.7398529
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1966629, 45.1974106
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5239868, 59.5226593
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0543671, 56.0559921
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4224701, 38.4227715
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8303299, 45.8295288
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3392334, 55.3390427
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5926743, 47.5936203
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8421783, 44.8425751
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4022827, 49.4014969
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7926331, 60.7938004
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6380844, 59.6366196
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9780884, 57.9766312
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1694

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 718

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1999020, upper bound: 20.1416797
time: 65.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1413934, upper bound: 20.1686124
time: 54.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4054337, 38.4055099
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9533997, 41.9520988
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1127472, 36.1108322
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5971680, 38.5949326
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7396011, 50.7336655
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1944656, 45.2005081
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5122833, 59.5263748
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0482330, 56.0565033
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4218903, 38.4226608
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8248749, 45.8286438
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3278809, 55.3379669
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5829620, 47.5912781
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8328094, 44.8431015
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3926926, 49.4028778
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7893372, 60.7950897
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6264572, 59.6394653
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9633789, 57.9778137
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1559

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2003224, upper bound: 20.1674669
time: 62.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2012939, upper bound: 20.1664947
time: 56.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4060059, 38.4049454
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9531326, 41.9523659
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1126480, 36.1109314
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5969238, 38.5951729
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7374039, 50.7358627
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1949539, 45.2000351
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5171356, 59.5215149
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0500031, 56.0547447
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222717, 38.4222794
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8267365, 45.8267975
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3315125, 55.3343201
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5850983, 47.5891342
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8355255, 44.8403854
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3957748, 49.3997803
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7898865, 60.7945480
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6302414, 59.6356812
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9687042, 57.9724884
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1851169, upper bound: 20.1831723
time: 70.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1858201, upper bound: 20.1823743
time: 54.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 127.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1728273, upper bound: 20.1967525
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1726687, upper bound: 20.1968987
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1695033, upper bound: 20.1876464
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1608541, upper bound: 20.1962885
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1191002, upper bound: 20.1294592
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1191002, upper bound: 20.1294592
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1806733, upper bound: 20.1875768
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1771301, upper bound: 20.1911133
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1850301, upper bound: 20.1809879
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1850301, upper bound: 20.1809879
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1999020, upper bound: 20.1416797
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1413934, upper bound: 20.1686124
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.2003224, upper bound: 20.1674669
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.2012939, upper bound: 20.1664947
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1851169, upper bound: 20.1831723
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 127.32
Output dim: 5, lower bound: -20.1858201, upper bound: 20.1823743

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4033432, 38.4049454
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9542847, 41.9558487
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1065445, 36.1087036
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5967178, 38.5994186
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7478561, 50.7498856
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2066116, 45.2002411
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5511475, 59.5458527
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0623932, 56.0525589
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4224930, 38.4215546
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8327942, 45.8329620
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3434906, 55.3402481
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5810394, 47.5762939
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8572617, 44.8508682
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4207306, 49.4159164
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7985077, 60.7914581
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6608887, 59.6545334
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9972382, 57.9926071
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1387

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 720

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1709908, upper bound: 20.1949054
time: 44.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1709272, upper bound: 20.1949116
time: 63.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4034805, 38.4048157
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9542999, 41.9558372
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1065292, 36.1087189
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5967560, 38.5993805
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7470016, 50.7507248
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2068253, 45.2000275
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5534668, 59.5435486
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0623322, 56.0526276
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222946, 38.4217529
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8335114, 45.8322525
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3445129, 55.3392334
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5811310, 47.5762100
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8579636, 44.8501701
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4223785, 49.4142609
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7984772, 60.7914658
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6632385, 59.6521683
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9996185, 57.9902496
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 967

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1714757, upper bound: 20.1801348
time: 64.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1559007, upper bound: 20.1957080
time: 244.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4033508, 38.4051208
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9542770, 41.9559326
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1064072, 36.1088867
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5965271, 38.5998840
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7478409, 50.7506180
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2069473, 45.1997833
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5533295, 59.5467072
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0618668, 56.0513649
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4228210, 38.4217911
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8329773, 45.8330917
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3443298, 55.3389816
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5822449, 47.5749664
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8583221, 44.8500519
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4215851, 49.4154053
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7988586, 60.7912674
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6621094, 59.6542892
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9980927, 57.9923019
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1587

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1663853, upper bound: 20.1873268
time: 80.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1691833, upper bound: 20.1845312
time: 60.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4036331, 38.4048462
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9543686, 41.9558487
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1064224, 36.1088715
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5967255, 38.5996857
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7472458, 50.7512054
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2069473, 45.1997910
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5550537, 59.5449982
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0613174, 56.0519028
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4226074, 38.4220047
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8337097, 45.8323669
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3447876, 55.3385162
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5823212, 47.5748901
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8586197, 44.8497505
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4222565, 49.4147263
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7987213, 60.7913971
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6632690, 59.6531372
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0001068, 57.9903030
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 971

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 545

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1546798, upper bound: 20.1916719
time: 56.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1562470, upper bound: 20.1901061
time: 51.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4028320, 38.4021149
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9548111, 41.9546356
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1065750, 36.1069565
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5967789, 38.5972366
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7480621, 50.7496796
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2018814, 45.2010193
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5387344, 59.5358124
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0578232, 56.0590630
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4218369, 38.4219246
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8340836, 45.8325424
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3428955, 55.3386917
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5779495, 47.5735245
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8509445, 44.8479576
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4162292, 49.4146729
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7928162, 60.7927475
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6565704, 59.6554565
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9949799, 57.9914856
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1579258, upper bound: 20.1723206
time: 53.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1579184, upper bound: 20.1723279
time: 49.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4028015, 38.4021454
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9548111, 41.9546432
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1065826, 36.1069489
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5967331, 38.5972862
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7480621, 50.7496643
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2018661, 45.2010422
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5386429, 59.5358963
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0578537, 56.0590401
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4219284, 38.4218407
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8340683, 45.8325577
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3428345, 55.3387680
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5779953, 47.5734711
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8509064, 44.8479996
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4161224, 49.4147720
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7928467, 60.7927017
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6562958, 59.6557236
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9949036, 57.9915619
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 545

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1710338, upper bound: 20.1867807
time: 56.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1727913, upper bound: 20.1850266
time: 65.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4052963, 38.4049606
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9529953, 41.9526978
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1127243, 36.1123123
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5972137, 38.5962982
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7382660, 50.7383957
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1961746, 45.1971283
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5223694, 59.5221939
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0544281, 56.0562897
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4224014, 38.4228134
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8294754, 45.8288879
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3375702, 55.3378983
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5921936, 47.5940704
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8405151, 44.8414459
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3989410, 49.3989868
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7917938, 60.7930756
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6332321, 59.6327896
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9744720, 57.9739075
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 532

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1588

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1846523, upper bound: 20.1808071
time: 72.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1848501, upper bound: 20.1806094
time: 58.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4052277, 38.4050293
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9529266, 41.9527702
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1126480, 36.1123924
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5971909, 38.5963249
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7379608, 50.7386932
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1965103, 45.1967850
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5237122, 59.5208511
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0548859, 56.0558167
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4223785, 38.4228325
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8296280, 45.8287430
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3378143, 55.3376541
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5922089, 47.5940628
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8408356, 44.8411255
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4000854, 49.3978424
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7920837, 60.7928009
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6349869, 59.6310349
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9754791, 57.9729004
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1718

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1197724, upper bound: 20.1098455
time: 53.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1138591, upper bound: 20.1157672
time: 49.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4051056, 38.4040146
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9529419, 41.9521332
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1124496, 36.1114731
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5969467, 38.5957489
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7386246, 50.7381744
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1947403, 45.1977539
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5237579, 59.5254669
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0524902, 56.0576172
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4224319, 38.4227676
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8301773, 45.8293686
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3378906, 55.3387680
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5918350, 47.5933380
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8392792, 44.8415718
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3976440, 49.3995590
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7894745, 60.7929535
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6310806, 59.6337814
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9755249, 57.9765778
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1585

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 561

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1902982, upper bound: 20.1321494
time: 64.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1902982, upper bound: 20.1321494
time: 62.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4053116, 38.4054108
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9533310, 41.9520454
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1135330, 36.1116180
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5988770, 38.5965729
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7425156, 50.7366257
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1962280, 45.2022247
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5140686, 59.5280304
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0466003, 56.0547867
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222107, 38.4229965
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8272247, 45.8309631
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3343811, 55.3444977
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5904388, 47.5990067
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8393250, 44.8496437
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3959351, 49.4058609
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7918091, 60.7974396
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6296463, 59.6423340
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9696655, 57.9840317
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1935608, upper bound: 20.1672462
time: 51.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1665604, upper bound: 20.1607225
time: 48.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4053345, 38.4053841
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9533463, 41.9520340
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1135330, 36.1116142
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5988083, 38.5966454
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7425461, 50.7365799
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1961823, 45.2022781
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5139313, 59.5281601
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0465240, 56.0548630
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222260, 38.4229813
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8271942, 45.8309937
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3344116, 55.3444824
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5906830, 47.5987549
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8393402, 44.8496246
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3956757, 49.4061127
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7916870, 60.7975388
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6293411, 59.6426315
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9696045, 57.9841080
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1358

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1466990, upper bound: 20.1664742
time: 47.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2012729, upper bound: 20.1454388
time: 56.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4071045, 38.4058762
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9534454, 41.9524727
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1137848, 36.1117935
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5965881, 38.5946693
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7377319, 50.7354584
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1855240, 45.1916275
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.4984589, 59.5049286
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0415039, 56.0475883
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222260, 38.4222260
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8249664, 45.8254242
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3239365, 55.3276291
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5885773, 47.5931778
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8333359, 44.8392563
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3903046, 49.3959579
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7867737, 60.7923965
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6281128, 59.6359711
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9505539, 57.9563828
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1637

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1448

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1831763, upper bound: 20.1812486
time: 45.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1831928, upper bound: 20.1812322
time: 46.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4069366, 38.4060440
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9532394, 41.9526825
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1135178, 36.1120605
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5964279, 38.5948410
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7369995, 50.7361526
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1864853, 45.1906052
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5003510, 59.5028229
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0428314, 56.0462494
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4222260, 38.4222298
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8253632, 45.8250351
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3248215, 55.3267593
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5891724, 47.5926132
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8343964, 44.8381996
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.3918610, 49.3943100
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7877045, 60.7914505
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6303711, 59.6335678
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9523544, 57.9543152
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1646

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1776

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1856340, upper bound: 20.1821345
time: 62.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1855800, upper bound: 20.1821885
time: 71.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 136.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1709908, upper bound: 20.1949054
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1709272, upper bound: 20.1949116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1714757, upper bound: 20.1801348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1559007, upper bound: 20.1957080
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1663853, upper bound: 20.1873268
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1691833, upper bound: 20.1845312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1546798, upper bound: 20.1916719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1562470, upper bound: 20.1901061
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1579258, upper bound: 20.1723206
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1579184, upper bound: 20.1723279
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1710338, upper bound: 20.1867807
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1727913, upper bound: 20.1850266
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1846523, upper bound: 20.1808071
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1848501, upper bound: 20.1806094
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1197724, upper bound: 20.1098455
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1138591, upper bound: 20.1157672
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1902982, upper bound: 20.1321494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1902982, upper bound: 20.1321494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1935608, upper bound: 20.1672462
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1665604, upper bound: 20.1607225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1466990, upper bound: 20.1664742
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.2012729, upper bound: 20.1454388
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1831763, upper bound: 20.1812486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1831928, upper bound: 20.1812322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1856340, upper bound: 20.1821345
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 136.08
Output dim: 5, lower bound: -20.1855800, upper bound: 20.1821885

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4028397, 38.4051132
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9538116, 41.9556580
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1064529, 36.1088028
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5964890, 38.6000481
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7462311, 50.7465057
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2064667, 45.2000427
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5434570, 59.5430527
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0597305, 56.0480728
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
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4221649, 38.4205551
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8294220, 45.8313980
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3400726, 55.3385239
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5803070, 47.5762177
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8558960, 44.8500443
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4143066, 49.4128113
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7994537, 60.7913208
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6516113, 59.6500702
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9874115, 57.9879303
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1448

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1638

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1683164, upper bound: 20.1912858
time: 52.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1673702, upper bound: 20.1922386
time: 150.94 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 206.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 206.10
Output dim: 5, lower bound: -20.1683164, upper bound: 20.1912858
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 206.10
Output dim: 5, lower bound: -20.1673702, upper bound: 20.1922386
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1709272, upper bound: 20.1949116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1559007, upper bound: 20.1957080
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1663853, upper bound: 20.1873268
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1691833, upper bound: 20.1845312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1546798, upper bound: 20.1916719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1562470, upper bound: 20.1901061
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1710338, upper bound: 20.1867807
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1727913, upper bound: 20.1850266
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1846523, upper bound: 20.1808071
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1848501, upper bound: 20.1806094
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1902982, upper bound: 20.1321494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1902982, upper bound: 20.1321494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1935608, upper bound: 20.1672462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.2012729, upper bound: 20.1454388
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1856340, upper bound: 20.1821345
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 206.10
Output dim: 5, lower bound: -20.1855800, upper bound: 20.1821885

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 59.31 + 3652.83 = 3712.15 seconds
