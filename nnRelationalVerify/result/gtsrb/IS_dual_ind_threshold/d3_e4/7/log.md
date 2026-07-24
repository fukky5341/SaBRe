## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 7)
Time budget: 7200 seconds
Split limit: 100
Threshold: 97.2066343617


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-125.3283386, 84.5538406, -125.3283386, 84.5538406, -209.8821716, 209.8821716)
1: (-70.4384460, 74.4370880, -70.4384460, 74.4370880, -144.8755341, 144.8755341)
2: (-63.3939476, 71.4393845, -63.3939476, 71.4393845, -134.8333282, 134.8333282)
3: (-72.9958878, 86.5009003, -72.9958878, 86.5009003, -159.4967957, 159.4967957)
4: (-76.0572510, 84.7594757, -76.0572510, 84.7594757, -160.8167267, 160.8167267)
5: (-68.1720428, 90.8613434, -68.1720428, 90.8613434, -159.0333862, 159.0333710)
6: (-102.8753433, 76.1697769, -102.8753433, 76.1697769, -179.0451202, 179.0451202)
7: (-84.0719299, 91.3842773, -84.0719299, 91.3842773, -175.4562073, 175.4561920)
8: (-89.2355499, 101.8614655, -89.2355499, 101.8614655, -191.0970154, 191.0970154)
9: (-78.5874634, 82.0120239, -78.5874634, 82.0120239, -160.5994873, 160.5994873)
10: (-111.4125748, 118.6578827, -111.4125748, 118.6578827, -230.0704651, 230.0704498)
11: (-111.1244888, 84.4927597, -111.1244888, 84.4927597, -195.6172485, 195.6172485)
12: (-111.4331055, 89.9215240, -111.4331055, 89.9215240, -201.3546295, 201.3546295)
13: (-110.7758636, 100.7236710, -110.7758636, 100.7236710, -211.4995270, 211.4995117)
14: (-163.2827911, 84.5499115, -163.2827911, 84.5499115, -247.8326569, 247.8326874)
15: (-92.1589890, 81.8159943, -92.1589890, 81.8159943, -173.9749756, 173.9749756)
16: (-118.5491333, 97.9611740, -118.5491333, 97.9611740, -216.5103149, 216.5103149)
17: (-164.7108154, 120.6250458, -164.7108154, 120.6250458, -285.3358765, 285.3358765)
18: (-102.0499420, 85.4436569, -102.0499420, 85.4436569, -187.4935608, 187.4935608)
19: (-85.3727036, 48.0397758, -85.3727036, 48.0397758, -133.4124756, 133.4124756)
20: (-74.9602051, 57.8805199, -74.9602051, 57.8805199, -132.8406982, 132.8407135)
21: (-104.8095627, 63.8468246, -104.8095627, 63.8468246, -168.6563873, 168.6563721)
22: (-113.4339752, 73.5631409, -113.4339752, 73.5631409, -186.9971161, 186.9971161)
23: (-86.6149139, 58.8942108, -86.6149139, 58.8942108, -145.5091248, 145.5091248)
24: (-103.7554398, 69.6435852, -103.7554398, 69.6435852, -173.3990173, 173.3990021)
25: (-91.1081543, 68.4662781, -91.1081543, 68.4662781, -159.5744324, 159.5744324)
26: (-122.4749908, 90.5463257, -122.4749908, 90.5463257, -213.0213165, 213.0213165)
27: (-104.6645432, 74.4739075, -104.6645432, 74.4739075, -179.1384277, 179.1384583)
28: (-85.8082733, 63.4338379, -85.8082733, 63.4338379, -149.2421112, 149.2421112)
29: (-119.4852905, 77.4210129, -119.4852905, 77.4210129, -196.9063110, 196.9062958)
30: (-102.9461212, 80.2182465, -102.9461212, 80.2182465, -183.1643372, 183.1643372)
31: (-106.6771393, 67.5839539, -106.6771393, 67.5839539, -174.2610931, 174.2610931)
32: (-100.1806641, 73.7657471, -100.1806641, 73.7657471, -173.9464111, 173.9463959)
33: (-141.2260132, 80.9459229, -141.2260132, 80.9459229, -222.1719360, 222.1719360)
34: (-120.2172318, 73.0502930, -120.2172318, 73.0502930, -193.2674866, 193.2675018)
35: (-120.8067245, 70.4543076, -120.8067245, 70.4543076, -191.2610321, 191.2610321)
36: (-117.9640656, 69.8447800, -117.9640656, 69.8447800, -187.8088379, 187.8088379)
37: (-164.8734131, 74.2879410, -164.8734131, 74.2879410, -239.1613464, 239.1613464)
38: (-145.9729004, 86.4679642, -145.9729004, 86.4679642, -232.4408569, 232.4408569)
39: (-168.6121521, 78.1188583, -168.6121521, 78.1188583, -246.7310181, 246.7310181)
40: (-135.6280518, 73.9228058, -135.6280518, 73.9228058, -209.5508575, 209.5508575)
41: (-100.8306885, 67.4778748, -100.8306885, 67.4778748, -168.3085632, 168.3085632)
42: (-75.8641663, 65.9938202, -75.8641663, 65.9938202, -141.8579865, 141.8579712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.00 + 116.31 = 119.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -97.3039383, upper bound: 97.3039383

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2499324, upper bound: 97.3022695
time: 108.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.3024426, upper bound: 97.3024427
time: 172.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 280.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 280.80
Output dim: 5, lower bound: -97.2499324, upper bound: 97.3022695
IS_A2, status: Status.UNKNOWN, split count: 1, time: 280.80
Output dim: 5, lower bound: -97.3024426, upper bound: 97.3024427

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -124.6066666, 84.2873840, -124.9231720, 84.4195557, -209.0262146, 209.2105408
1: -69.9321289, 74.2248077, -70.1502991, 74.3419724, -144.2741089, 144.3750916
2: -62.7293396, 71.1545944, -63.0056190, 71.3475342, -134.0768738, 134.1602173
3: -72.1667099, 86.0914154, -72.5091705, 86.3658142, -158.5325317, 158.6005707
4: -75.2612152, 84.4494247, -75.5957184, 84.6409149, -159.9021301, 160.0451355
5: -67.5027466, 90.4818497, -67.7862549, 90.7411804, -158.2439270, 158.2680969
6: -102.5512543, 75.7055969, -102.6827545, 75.9060516, -178.4572906, 178.3883514
7: -83.4754562, 91.1614227, -83.7336655, 91.2829437, -174.7583923, 174.8950806
8: -88.5101929, 101.5078583, -88.8101349, 101.7322922, -190.2424927, 190.3179779
9: -78.1663818, 81.6335449, -78.3579788, 81.8004684, -159.9668579, 159.9915161
10: -110.7386780, 117.2646561, -111.1812897, 117.8429260, -228.5815735, 228.4459534
11: -110.5426636, 83.1887283, -110.9406586, 83.7077942, -194.2504578, 194.1293793
12: -110.9254379, 88.7152863, -111.2832489, 89.2170258, -200.1424561, 199.9985352
13: -109.8729401, 100.1326675, -110.2452316, 100.4712067, -210.3441315, 210.3778992
14: -162.6055756, 83.4861450, -163.0047455, 83.9201508, -246.5257111, 246.4908905
15: -91.4326019, 81.5452423, -91.7510071, 81.6459351, -173.0785217, 173.2962341
16: -118.0348358, 97.1803436, -118.2566452, 97.5047455, -215.5395813, 215.4369812
17: -164.1159363, 119.0831833, -164.5087280, 119.7077026, -283.8236389, 283.5919189
18: -101.4866257, 84.3787079, -101.7994003, 84.8154984, -186.3021088, 186.1781006
19: -84.9405899, 47.4515610, -85.2259598, 47.6883698, -132.6289673, 132.6775208
20: -74.5630646, 57.3862228, -74.7972031, 57.5886726, -132.1517334, 132.1834259
21: -104.2998657, 62.9968300, -104.6472626, 63.3420029, -167.6418610, 167.6440735
22: -113.0589981, 72.8634033, -113.2481842, 73.1490555, -186.2080536, 186.1115875
23: -86.2505188, 58.2469292, -86.4806061, 58.5144501, -144.7649689, 144.7275391
24: -103.3558960, 69.1258240, -103.5808640, 69.3379669, -172.6938629, 172.7066956
25: -90.8022842, 67.9390488, -90.9711761, 68.1536255, -158.9559021, 158.9102173
26: -121.9121246, 89.4086609, -122.2773972, 89.8860397, -211.7981262, 211.6860657
27: -104.2366867, 73.9641800, -104.4243317, 74.1748352, -178.4115143, 178.3884888
28: -85.4963226, 63.0298767, -85.6646271, 63.2006683, -148.6969910, 148.6945038
29: -119.1377106, 76.5448532, -119.3299026, 76.9007111, -196.0384064, 195.8747559
30: -102.5404358, 79.3659515, -102.7869873, 79.7212067, -182.2616425, 182.1529388
31: -106.1160583, 66.8169556, -106.4663239, 67.1240997, -173.2401428, 173.2832794
32: -99.8150787, 73.3589783, -99.9853973, 73.5321198, -173.3471985, 173.3443604
33: -140.3820343, 80.5549545, -140.7369232, 80.8017578, -221.1837463, 221.2918701
34: -119.5900650, 72.7058487, -119.8591919, 72.8801727, -192.4702301, 192.5650330
35: -120.0107956, 70.1563034, -120.3416977, 70.3369751, -190.3477478, 190.4980011
36: -117.3122711, 69.6056137, -117.5819626, 69.7298279, -187.0420990, 187.1875610
37: -164.3631897, 73.8990097, -164.5925903, 74.0781708, -238.4413605, 238.4915771
38: -145.1600647, 86.1004486, -145.5035095, 86.3188248, -231.4788666, 231.6039429
39: -167.7764587, 77.8355713, -168.1285095, 78.0059433, -245.7823792, 245.9640656
40: -135.0021057, 73.6946869, -135.2711334, 73.8213196, -208.8234253, 208.9657898
41: -100.4634705, 67.0853043, -100.6291809, 67.2690125, -167.7324829, 167.7144775
42: -75.5640106, 65.2241974, -75.7121582, 65.5519104, -141.1159210, 140.9363251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2453988, upper bound: 97.2439578
time: 104.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2461684, upper bound: 97.2991105
time: 104.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.2836914, 84.5387802, -125.3028488, 84.5451355, -209.8288269, 209.8416138
1: -70.4095001, 74.4239502, -70.4218140, 74.4295807, -144.8390808, 144.8457642
2: -63.3581696, 71.4284592, -63.3734131, 71.4330978, -134.7912598, 134.8018799
3: -72.9513702, 86.4834442, -72.9705353, 86.4908981, -159.4422455, 159.4539795
4: -76.0163956, 84.7425385, -76.0337296, 84.7497253, -160.7661133, 160.7762756
5: -68.1348724, 90.8475342, -68.1508789, 90.8534088, -158.9882812, 158.9984131
6: -102.8498077, 76.0811539, -102.8607330, 76.1199646, -178.9697723, 178.9418793
7: -84.0359802, 91.3690567, -84.0512772, 91.3756409, -175.4116211, 175.4203339
8: -89.1975403, 101.8446808, -89.2136230, 101.8518829, -191.0494080, 191.0583038
9: -78.5486755, 81.9712906, -78.5655060, 81.9886551, -160.5373230, 160.5368042
10: -111.3890762, 118.5826645, -111.3990097, 118.6146011, -230.0036774, 229.9816742
11: -111.1012955, 84.4298019, -111.1109695, 84.4556122, -195.5569000, 195.5407715
12: -111.4130249, 89.8627014, -111.4216843, 89.8877258, -201.3007507, 201.2843781
13: -110.7405396, 100.6908951, -110.7526855, 100.7049561, -211.4454498, 211.4435730
14: -163.2510681, 84.5008774, -163.2645569, 84.5219650, -247.7730408, 247.7654266
15: -92.0728302, 81.7929382, -92.1092606, 81.8027191, -173.8755493, 173.9021912
16: -118.5161896, 97.8834991, -118.5300446, 97.9138336, -216.4300232, 216.4135437
17: -164.6907959, 120.5500946, -164.6991882, 120.5821457, -285.2729187, 285.2492676
18: -102.0187836, 85.3914032, -102.0321350, 85.4140320, -187.4328156, 187.4235382
19: -85.3569641, 48.0078011, -85.3636322, 48.0215302, -133.3784943, 133.3714294
20: -74.9406281, 57.8567352, -74.9489288, 57.8669357, -132.8075562, 132.8056641
21: -104.7878342, 63.8062553, -104.7970276, 63.8237000, -168.6115417, 168.6032715
22: -113.3832321, 73.5227127, -113.4052734, 73.5399323, -186.9231567, 186.9279785
23: -86.5999985, 58.8614655, -86.6063385, 58.8755035, -145.4754944, 145.4678040
24: -103.7299271, 69.6165771, -103.7408142, 69.6283722, -173.3582764, 173.3573914
25: -91.0931015, 68.4360886, -91.0994492, 68.4490204, -159.5421143, 159.5355377
26: -122.4458618, 90.4897308, -122.4583588, 90.5140686, -212.9599304, 212.9480591
27: -104.6329498, 74.4486694, -104.6462860, 74.4596634, -179.0926208, 179.0949402
28: -85.7934875, 63.4117355, -85.7997284, 63.4208221, -149.2143097, 149.2114563
29: -119.4589386, 77.3727341, -119.4701767, 77.3934250, -196.8523560, 196.8429108
30: -102.9253006, 80.1761322, -102.9339752, 80.1943054, -183.1195984, 183.1101074
31: -106.6536560, 67.5429840, -106.6637115, 67.5607071, -174.2143555, 174.2066956
32: -100.1512756, 73.7404404, -100.1638184, 73.7507629, -173.9020386, 173.9042664
33: -141.1815643, 80.9286804, -141.2007141, 80.9359589, -222.1175232, 222.1293793
34: -120.1827469, 73.0269165, -120.1975403, 73.0368576, -193.2196045, 193.2244568
35: -120.7642517, 70.4415741, -120.7824707, 70.4469452, -191.2111969, 191.2240295
36: -117.9226379, 69.8307648, -117.9405365, 69.8367462, -187.7593842, 187.7713013
37: -164.8338623, 74.2456055, -164.8507080, 74.2637329, -239.0975952, 239.0962830
38: -145.9229126, 86.4525146, -145.9443970, 86.4589996, -232.3819122, 232.3968964
39: -168.5646667, 78.1052399, -168.5850525, 78.1110611, -246.6757202, 246.6902771
40: -135.5876770, 73.8635330, -135.6048126, 73.8892212, -209.4768982, 209.4683533
41: -100.8066711, 67.4122086, -100.8168564, 67.4406586, -168.2473145, 168.2290649
42: -75.8448944, 65.9455109, -75.8531189, 65.9661331, -141.8110352, 141.7986298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2991863, upper bound: 97.2444423
time: 111.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2993029, upper bound: 97.2993028
time: 111.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 225.93 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 225.93
Output dim: 5, lower bound: -97.2453988, upper bound: 97.2439578
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 225.93
Output dim: 5, lower bound: -97.2461684, upper bound: 97.2991105
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 225.93
Output dim: 5, lower bound: -97.2991863, upper bound: 97.2444423
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 225.93
Output dim: 5, lower bound: -97.2993029, upper bound: 97.2993028

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -124.4452972, 84.2441101, -124.6231461, 84.3339386, -208.7792358, 208.8672485
1: -69.8247986, 74.1950378, -69.9478455, 74.2855988, -144.1103973, 144.1428833
2: -62.5605049, 71.1245499, -62.6895676, 71.2911987, -133.8516998, 133.8141174
3: -71.9713898, 86.0496597, -72.1420059, 86.2879028, -158.2592926, 158.1916504
4: -75.0697021, 84.4145966, -75.2435074, 84.5703125, -159.6399994, 159.6581116
5: -67.3296509, 90.4447937, -67.4666138, 90.6715546, -158.0012054, 157.9114075
6: -102.4913406, 75.6049423, -102.5658493, 75.7182007, -178.2095337, 178.1707764
7: -83.3312988, 91.1295929, -83.4622498, 91.2231979, -174.5545044, 174.5918427
8: -88.3388672, 101.4673080, -88.4881592, 101.6567917, -189.9956360, 189.9554749
9: -78.0684967, 81.5628204, -78.1609344, 81.6681061, -159.7366028, 159.7237549
10: -110.6610260, 117.0289536, -111.0244827, 117.4090042, -228.0700073, 228.0534058
11: -110.4770660, 82.8774643, -110.8178177, 83.1361160, -193.6131897, 193.6952667
12: -110.8771820, 88.4254074, -111.1884460, 88.6857376, -199.5629272, 199.6138611
13: -109.7034149, 100.0409546, -109.9182816, 100.3024445, -210.0058441, 209.9592285
14: -162.5025635, 83.2545929, -162.8117065, 83.4818420, -245.9844055, 246.0662994
15: -91.2972183, 81.4852753, -91.4975052, 81.5307312, -172.8279419, 172.9827728
16: -117.9346695, 97.0418091, -118.0607452, 97.2508774, -215.1855469, 215.1025543
17: -164.0357361, 118.7235184, -164.3576050, 119.0252457, -283.0609741, 283.0811157
18: -101.4020081, 84.1444550, -101.6433563, 84.3574753, -185.7594910, 185.7878113
19: -84.8883972, 47.3074379, -85.1296692, 47.4147987, -132.3031769, 132.4371033
20: -74.5070038, 57.2673836, -74.6922150, 57.3616791, -131.8686829, 131.9595947
21: -104.2417984, 62.7771759, -104.5387115, 62.9334488, -167.1752472, 167.3158722
22: -112.9999847, 72.6767273, -113.1365662, 72.7863388, -185.7863159, 185.8132935
23: -86.2022095, 58.1055069, -86.3914032, 58.2504349, -144.4526367, 144.4969177
24: -103.2984390, 69.0091400, -103.4760971, 69.1064148, -172.4048462, 172.4852295
25: -90.7559967, 67.8225174, -90.8851547, 67.9230347, -158.6790314, 158.7076721
26: -121.8470917, 89.1203308, -122.1548462, 89.3464737, -211.1935730, 211.2751617
27: -104.1539764, 73.8119049, -104.2720642, 73.8779907, -178.0319672, 178.0839691
28: -85.4446487, 62.9171219, -85.5686493, 62.9807243, -148.4253693, 148.4857635
29: -119.0845795, 76.2864990, -119.2306900, 76.4158783, -195.5004272, 195.5171814
30: -102.4860535, 79.1583176, -102.6859665, 79.3307800, -181.8168182, 181.8442841
31: -106.0415192, 66.6720047, -106.3315582, 66.8394699, -172.8809814, 173.0035400
32: -99.7534103, 73.2379913, -99.8640900, 73.3105240, -173.0639343, 173.1020813
33: -140.1957550, 80.4981079, -140.3922577, 80.6935883, -220.8893127, 220.8903656
34: -119.4746780, 72.6377716, -119.6447525, 72.7427673, -192.2174377, 192.2825317
35: -119.8739548, 70.1176529, -120.0908127, 70.2593384, -190.1333008, 190.2084503
36: -117.2246857, 69.5590057, -117.4186630, 69.6380386, -186.8627319, 186.9776611
37: -164.2810364, 73.7919159, -164.4355164, 73.8805237, -238.1615601, 238.2274323
38: -145.0131226, 86.0450287, -145.2295227, 86.2076874, -231.2208099, 231.2745361
39: -167.6115723, 77.7932053, -167.8227539, 77.9284134, -245.5399628, 245.6159668
40: -134.9038086, 73.6414719, -135.0804138, 73.7230072, -208.6268158, 208.7218628
41: -100.4008331, 66.9852600, -100.5034943, 67.0851593, -167.4859619, 167.4887543
42: -75.5200195, 65.0553360, -75.6163406, 65.2432404, -140.7632294, 140.6716766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.1896325
time: 101.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2356240
time: 107.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -124.5710602, 84.2745895, -124.9552155, 84.5178833, -209.0889435, 209.2297974
1: -69.9072800, 74.2166748, -70.1608276, 74.4303589, -144.3376312, 144.3775024
2: -62.6941948, 71.1458435, -62.9890480, 71.6002045, -134.2943726, 134.1348877
3: -72.1279907, 86.0779724, -72.4935379, 86.6681366, -158.7961273, 158.5715027
4: -75.2207642, 84.4386139, -75.5842514, 84.8501129, -160.0708771, 160.0228577
5: -67.4688797, 90.4705048, -67.7808914, 91.0635681, -158.5324402, 158.2514038
6: -102.5329971, 75.6209793, -102.7584915, 75.8579407, -178.3909302, 178.3794708
7: -83.4393616, 91.1511993, -83.7436447, 91.3915787, -174.8309326, 174.8948364
8: -88.4745407, 101.4948273, -88.8026123, 101.9582062, -190.4327393, 190.2974396
9: -78.1428146, 81.6031342, -78.3792114, 81.8416595, -159.9844666, 159.9823456
10: -110.7194824, 117.2123718, -111.3396835, 117.8796387, -228.5990906, 228.5520630
11: -110.5166779, 83.1297760, -111.3220367, 83.6639786, -194.1806641, 194.4517822
12: -110.9112396, 88.6579971, -111.7101288, 89.2021179, -200.1133423, 200.3681183
13: -109.7899780, 100.1041183, -110.1856689, 100.6840363, -210.4739990, 210.2897949
14: -162.5788269, 83.4453583, -163.2460632, 83.8934555, -246.4722748, 246.6914062
15: -91.3247223, 81.5236969, -91.7098465, 81.7259979, -173.0507202, 173.2335358
16: -118.0058899, 97.0573425, -118.3787766, 97.4184265, -215.4243164, 215.4361267
17: -164.0960999, 119.0120087, -164.8848724, 119.6484146, -283.7445068, 283.8968811
18: -101.4621048, 84.3336029, -102.0867310, 84.7882233, -186.2503204, 186.4203186
19: -84.9256821, 47.4237404, -85.4938812, 47.6697693, -132.5954285, 132.9176178
20: -74.5458527, 57.3641891, -74.9700012, 57.5821648, -132.1279907, 132.3341675
21: -104.2807236, 62.9589043, -104.9939575, 63.3153992, -167.5961304, 167.9528656
22: -113.0392685, 72.8228760, -113.3653107, 73.1260529, -186.1653137, 186.1881866
23: -86.2385101, 58.2193489, -86.6920319, 58.5085182, -144.7470245, 144.9113617
24: -103.3375320, 69.1065903, -103.7321014, 69.3276749, -172.6651917, 172.8386841
25: -90.7866974, 67.9109650, -91.0502319, 68.1352005, -158.9219055, 158.9611816
26: -121.8899078, 89.3552170, -122.6890259, 89.8638153, -211.7537231, 212.0442505
27: -104.2114258, 73.9385376, -104.6004791, 74.1582565, -178.3696899, 178.5390167
28: -85.4840546, 63.0076218, -85.8641586, 63.1999817, -148.6840363, 148.8717804
29: -119.1194992, 76.4951019, -119.5084305, 76.8540878, -195.9735870, 196.0035095
30: -102.5197144, 79.3264008, -103.0052338, 79.7191086, -182.2388153, 182.3316345
31: -106.0968628, 66.7886047, -106.7466125, 67.1000824, -173.1969452, 173.5352173
32: -99.7960510, 73.3339844, -100.0696945, 73.5495529, -173.3455658, 173.4036865
33: -140.3439331, 80.5380096, -140.7416077, 80.9688263, -221.3127594, 221.2796173
34: -119.5659866, 72.6850891, -119.8932724, 72.9359589, -192.5018921, 192.5783386
35: -119.9745789, 70.1449585, -120.3496399, 70.4248123, -190.3993835, 190.4945984
36: -117.2833710, 69.5919189, -117.6139603, 69.7654037, -187.0487366, 187.2058716
37: -164.3386841, 73.8549957, -164.6949158, 74.0769806, -238.4156647, 238.5499115
38: -145.1240540, 86.0744400, -145.5656128, 86.3668671, -231.4908752, 231.6400452
39: -167.7365112, 77.8232880, -168.1520996, 78.1538849, -245.8903961, 245.9753876
40: -134.9742432, 73.6309433, -135.3065491, 73.8099670, -208.7842102, 208.9375000
41: -100.4468689, 67.0156403, -100.6979294, 67.2435913, -167.6904449, 167.7135620
42: -75.5511017, 65.1887741, -75.8158722, 65.5779800, -141.1290588, 141.0046387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2459027
time: 130.98 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2920153
time: 93.87 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.1141052, 84.4901733, -124.9951935, 84.4573975, -209.5715027, 209.4853516
1: -70.2946472, 74.3920898, -70.2139053, 74.3718567, -144.6665039, 144.6059875
2: -63.1819000, 71.3966217, -63.0533943, 71.3754120, -134.5572968, 134.4499969
3: -72.7468872, 86.4387970, -72.5989532, 86.4100266, -159.1569214, 159.0377502
4: -75.8198090, 84.7019577, -75.6767349, 84.6763000, -160.4960938, 160.3786926
5: -67.9568939, 90.8081970, -67.8275681, 90.7821350, -158.7390137, 158.6357727
6: -102.7824173, 75.9712143, -102.7387238, 75.9232101, -178.7056274, 178.7099152
7: -83.8809204, 91.3353653, -83.7706909, 91.3145294, -175.1954498, 175.1060486
8: -89.0169144, 101.8018799, -88.8859100, 101.7743683, -190.7912750, 190.6877899
9: -78.4349518, 81.8955231, -78.3598862, 81.8517761, -160.2867126, 160.2554016
10: -111.2989731, 118.3404465, -111.2361526, 118.1754150, -229.4743958, 229.5765991
11: -111.0313568, 84.1135483, -110.9844742, 83.8801575, -194.9114990, 195.0980225
12: -111.3593979, 89.5677872, -111.3243561, 89.3516083, -200.7109985, 200.8921509
13: -110.5469513, 100.5943146, -110.4092102, 100.5298157, -211.0767670, 211.0035095
14: -163.1429749, 84.2568817, -163.0684204, 84.0779648, -247.2209473, 247.3253021
15: -91.9227371, 81.7261581, -91.8374786, 81.6819458, -173.6046753, 173.5636292
16: -118.4035263, 97.7381210, -118.3262939, 97.6532211, -216.0567474, 216.0643921
17: -164.6058502, 120.1702347, -164.5451355, 119.8911133, -284.4969482, 284.7153625
18: -101.9276810, 85.1343384, -101.8671951, 84.9467545, -186.8744354, 187.0015106
19: -85.3016891, 47.8549500, -85.2635193, 47.7441521, -133.0458374, 133.1184692
20: -74.8811035, 57.7296181, -74.8411255, 57.6360779, -132.5171814, 132.5707397
21: -104.7260361, 63.5795212, -104.6852112, 63.4114990, -168.1375427, 168.2647400
22: -113.3214874, 73.3152313, -113.2924957, 73.1641083, -186.4855652, 186.6077271
23: -86.5498505, 58.7142639, -86.5153961, 58.6081276, -145.1579590, 145.2296600
24: -103.6701736, 69.4866943, -103.6325684, 69.3921661, -173.0623169, 173.1192627
25: -91.0443573, 68.3045502, -91.0113220, 68.2106934, -159.2550507, 159.3158569
26: -122.3761292, 90.1885834, -122.3319550, 89.9667816, -212.3428955, 212.5205383
27: -104.5461884, 74.2823029, -104.4889908, 74.1572189, -178.7033997, 178.7713013
28: -85.7396927, 63.2879257, -85.7020874, 63.1962967, -148.9359741, 148.9900055
29: -119.4029770, 77.1006775, -119.3686218, 76.8993530, -196.3023071, 196.4692841
30: -102.8677216, 79.9584961, -102.8298492, 79.7988129, -182.6665344, 182.7883453
31: -106.5771255, 67.3825760, -106.5250320, 67.2696533, -173.8467712, 173.9076080
32: -100.0826340, 73.6163483, -100.0391846, 73.5267181, -173.6093445, 173.6555176
33: -140.9899750, 80.8666611, -140.8523407, 80.8236465, -221.8136292, 221.7189941
34: -120.0622787, 72.9476471, -119.9788361, 72.8937683, -192.9560242, 192.9264832
35: -120.6220779, 70.3966141, -120.5273209, 70.3656158, -190.9877014, 190.9239349
36: -117.8305054, 69.7768250, -117.7739182, 69.7395782, -187.5700684, 187.5507507
37: -164.7441101, 74.1345749, -164.6879272, 74.0629578, -238.8070679, 238.8225098
38: -145.7678833, 86.3886566, -145.6634827, 86.3439789, -232.1118469, 232.0521393
39: -168.3932495, 78.0610352, -168.2738342, 78.0310364, -246.4242859, 246.3348694
40: -135.4781342, 73.8065262, -135.4056396, 73.7867126, -209.2648468, 209.2121582
41: -100.7342682, 67.3093567, -100.6854630, 67.2548141, -167.9890747, 167.9948120
42: -75.7890167, 65.7726440, -75.7520599, 65.6526184, -141.4416351, 141.5246887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.1905344
time: 104.53 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2371879
time: 131.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.2459183, 84.5247040, -125.3330994, 84.6412888, -209.8872070, 209.8578033
1: -70.3839035, 74.4152222, -70.4326553, 74.5174332, -144.9013214, 144.8478699
2: -63.3230820, 71.4194946, -63.3586044, 71.6853180, -135.0083923, 134.7780914
3: -72.9119568, 86.4688492, -72.9562302, 86.7913742, -159.7033386, 159.4250793
4: -75.9762421, 84.7303467, -76.0227432, 84.9580002, -160.9342346, 160.7530823
5: -68.1012497, 90.8358383, -68.1467133, 91.1750183, -159.2762756, 158.9825439
6: -102.8296204, 75.9946136, -102.9364624, 76.0669937, -178.8966064, 178.9310608
7: -83.9981995, 91.3582916, -84.0615845, 91.4835739, -175.4817657, 175.4198761
8: -89.1611786, 101.8312225, -89.2066193, 102.0771942, -191.2383728, 191.0378418
9: -78.5261230, 81.9393692, -78.5901794, 82.0285187, -160.5546265, 160.5295410
10: -111.3671341, 118.5322800, -111.5547256, 118.6542206, -230.0213470, 230.0870056
11: -111.0751419, 84.3720856, -111.4890747, 84.4137802, -195.4888916, 195.8611450
12: -111.3974915, 89.8054962, -111.8474426, 89.8730316, -201.2705231, 201.6529388
13: -110.6659241, 100.6597824, -110.7043686, 100.9160614, -211.5819855, 211.3641357
14: -163.2233582, 84.4571381, -163.5048523, 84.4948349, -247.7181702, 247.9619904
15: -91.9708328, 81.7696686, -92.0645752, 81.8819427, -173.8527832, 173.8342438
16: -118.4838409, 97.7605743, -118.6523438, 97.8276367, -216.3114777, 216.4129181
17: -164.6698608, 120.4751282, -165.0742188, 120.5235214, -285.1933899, 285.5493469
18: -101.9913712, 85.3413925, -102.3087463, 85.3871765, -187.3785400, 187.6501160
19: -85.3413544, 47.9797821, -85.6297302, 48.0049171, -133.3462677, 133.6095123
20: -74.9225311, 57.8334045, -75.1205826, 57.8607826, -132.7832947, 132.9539795
21: -104.7679443, 63.7683411, -105.1428833, 63.7985687, -168.5665131, 168.9112244
22: -113.3629532, 73.4780045, -113.5222778, 73.5173035, -186.8802490, 187.0002747
23: -86.5876694, 58.8340645, -86.8172302, 58.8704758, -145.4581451, 145.6512909
24: -103.7103729, 69.5941467, -103.8896408, 69.6181183, -173.3284912, 173.4837952
25: -91.0762939, 68.4050293, -91.1782455, 68.4307861, -159.5070801, 159.5832825
26: -122.4219208, 90.4345551, -122.8676605, 90.4923477, -212.9142761, 213.3022156
27: -104.6061249, 74.4197845, -104.8192520, 74.4430313, -179.0491486, 179.2390442
28: -85.7808838, 63.3880653, -85.9984512, 63.4212036, -149.2020721, 149.3865204
29: -119.4398956, 77.3204193, -119.6475525, 77.3470306, -196.7869263, 196.9679718
30: -102.9041595, 80.1357346, -103.1513062, 80.1935425, -183.0977020, 183.2870483
31: -106.6332245, 67.5119553, -106.9397659, 67.5377350, -174.1709595, 174.4517059
32: -100.1307678, 73.7155151, -100.2464218, 73.7686615, -173.8994293, 173.9619141
33: -141.1427002, 80.9107666, -141.2050781, 81.1011200, -222.2438049, 222.1158447
34: -120.1571274, 73.0043488, -120.2302246, 73.0938568, -193.2509766, 193.2345734
35: -120.7291107, 70.4288330, -120.7918625, 70.5351257, -191.2642365, 191.2207031
36: -117.8932724, 69.8149261, -117.9730225, 69.8727417, -187.7660065, 187.7879486
37: -164.8069763, 74.2012634, -164.9507141, 74.2604523, -239.0674286, 239.1519775
38: -145.8864441, 86.4242477, -146.0065308, 86.5077820, -232.3942261, 232.4307709
39: -168.5246277, 78.0925446, -168.6088409, 78.2578888, -246.7824860, 246.7013855
40: -135.5554047, 73.8028870, -135.6371155, 73.8819504, -209.4373474, 209.4400024
41: -100.7876358, 67.3373642, -100.8833466, 67.4132156, -168.2008514, 168.2207031
42: -75.8291702, 65.9068069, -75.9548492, 65.9899445, -141.8191223, 141.8616638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=502, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2291158
time: 132.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2944053
time: 109.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 244.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.2363991, upper bound: 97.1896325
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2356240
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2459027
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2920153
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.2363991, upper bound: 97.1905344
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.2363991, upper bound: 97.2371879
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2291158
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 244.19
Output dim: 5, lower bound: -97.1985144, upper bound: 97.2944053

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -124.1892776, 83.9105377, -124.5630493, 84.1610641, -208.3503418, 208.4735870
1: -69.6410675, 73.9681244, -69.9116669, 74.1701355, -143.8112030, 143.8797760
2: -62.3512421, 70.8681183, -62.6541824, 71.1530991, -133.5043335, 133.5223083
3: -71.7191925, 85.6890564, -72.1083374, 86.0967865, -157.8159637, 157.7973938
4: -74.8923264, 84.2986145, -75.2014542, 84.5188293, -159.4111481, 159.5000305
5: -67.0500183, 90.0433807, -67.4249268, 90.4583282, -157.5083466, 157.4683075
6: -102.3365250, 75.2415543, -102.5189972, 75.5402145, -177.8767242, 177.7605591
7: -82.9663162, 90.6452332, -83.3988419, 90.9592285, -173.9255371, 174.0440674
8: -88.1237106, 101.1918488, -88.4564972, 101.5133057, -189.6370239, 189.6483459
9: -77.8306503, 81.3577576, -78.0433807, 81.6311035, -159.4617615, 159.4011383
10: -110.2807770, 116.6794739, -110.8335800, 117.3579254, -227.6387024, 227.5130310
11: -110.3220978, 82.6571198, -110.7331619, 83.0920563, -193.4141541, 193.3902283
12: -110.2112274, 87.9587860, -110.8205338, 88.6378250, -198.8490601, 198.7793274
13: -109.3291092, 99.7623291, -109.7393341, 100.2202606, -209.5493774, 209.5016479
14: -161.8848267, 82.9236755, -162.5031433, 83.4538269, -245.3386536, 245.4268188
15: -90.8051758, 81.2362976, -91.2665863, 81.4762726, -172.2814484, 172.5028839
16: -117.6656723, 96.7272186, -117.9742508, 97.1307755, -214.7964478, 214.7014465
17: -163.5808868, 118.3800507, -164.1249695, 118.9806137, -282.5614929, 282.5050049
18: -101.1624146, 84.0182877, -101.5371780, 84.3231506, -185.4855499, 185.5554657
19: -84.7686539, 47.2459183, -85.0720673, 47.3953896, -132.1640320, 132.3179626
20: -74.3256378, 57.1789055, -74.6162872, 57.3366661, -131.6622925, 131.7951813
21: -104.0973206, 62.6582184, -104.4594421, 62.9031448, -167.0004578, 167.1176605
22: -112.4563904, 72.3552399, -112.8568878, 72.7344666, -185.1908569, 185.2121277
23: -86.0682144, 57.9982224, -86.3340607, 58.2102356, -144.2784424, 144.3322754
24: -103.1353607, 68.9285812, -103.4178009, 69.0754395, -172.2108002, 172.3463593
25: -90.5241852, 67.6417160, -90.7716370, 67.8747864, -158.3989716, 158.4133453
26: -121.1219635, 88.7142944, -121.7756958, 89.3044357, -210.4263916, 210.4899597
27: -103.9134674, 73.6572571, -104.2111588, 73.8051605, -177.7186279, 177.8684082
28: -85.3073883, 62.8132019, -85.5210266, 62.9376793, -148.2450562, 148.3342285
29: -118.7031555, 75.9669800, -119.0362091, 76.3697510, -195.0728760, 195.0031891
30: -102.3288803, 78.9444427, -102.6268005, 79.2426605, -181.5715332, 181.5712280
31: -105.8760147, 66.5568237, -106.2633209, 66.7998428, -172.6758423, 172.8201294
32: -99.5573120, 73.0742188, -99.7748566, 73.2659912, -172.8232880, 172.8490601
33: -139.9716949, 80.3630219, -140.3388062, 80.6295090, -220.6011963, 220.7018280
34: -119.2343750, 72.4544067, -119.5886383, 72.6547852, -191.8891602, 192.0430298
35: -119.6734543, 69.9749146, -120.0426636, 70.1906281, -189.8640747, 190.0175781
36: -117.0413895, 69.4556198, -117.3504715, 69.5870819, -186.6284485, 186.8060913
37: -164.0395813, 73.6874084, -164.3423767, 73.8424225, -237.8820038, 238.0297852
38: -144.7912903, 85.9134903, -145.1744843, 86.1476898, -230.9389648, 231.0879669
39: -167.3741150, 77.6678238, -167.7354431, 77.8721542, -245.2462769, 245.4032593
40: -134.6870422, 73.3970184, -135.0249023, 73.5914459, -208.2784729, 208.4219208
41: -100.2521057, 66.7279510, -100.4591980, 66.9613495, -167.2134399, 167.1871338
42: -75.4393158, 64.8437271, -75.5726471, 65.1826935, -140.6220093, 140.4163818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1410031
time: 107.73 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1803784
time: 107.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -124.4213409, 84.2265396, -124.6100388, 84.3240509, -208.7453918, 208.8365784
1: -69.8106537, 74.1832733, -69.9400940, 74.2792664, -144.0899200, 144.1233521
2: -62.5461731, 71.1144562, -62.6816864, 71.2857285, -133.8319092, 133.7961426
3: -71.9579926, 86.0337982, -72.1346588, 86.2792816, -158.2372742, 158.1684570
4: -75.0547180, 84.4042816, -75.2352982, 84.5647125, -159.6194305, 159.6395874
5: -67.3175659, 90.4289932, -67.4598999, 90.6630783, -157.9806519, 157.8888855
6: -102.4763107, 75.5511703, -102.5576324, 75.6894073, -178.1656952, 178.1087952
7: -83.3140717, 91.1137390, -83.4527664, 91.2146988, -174.5287781, 174.5664978
8: -88.3239594, 101.4554520, -88.4799881, 101.6503143, -189.9742737, 189.9354401
9: -78.0540314, 81.5524445, -78.1529846, 81.6623459, -159.7163696, 159.7054291
10: -110.6409836, 117.0088043, -111.0135345, 117.3979263, -228.0388947, 228.0223083
11: -110.4547043, 82.7818909, -110.8055649, 83.0845413, -193.5392456, 193.5874481
12: -110.8547974, 88.4099426, -111.1764603, 88.6772690, -199.5320740, 199.5863800
13: -109.6619720, 100.0190506, -109.8952332, 100.2903519, -209.9523163, 209.9142761
14: -162.4771729, 83.2483063, -162.7980957, 83.4783707, -245.9555359, 246.0464020
15: -91.2342453, 81.4699554, -91.4637604, 81.5222244, -172.7564697, 172.9337158
16: -117.9097366, 96.9506454, -118.0469513, 97.2028503, -215.1125488, 214.9975891
17: -164.0189819, 118.6999435, -164.3485718, 119.0124969, -283.0314941, 283.0485229
18: -101.3867493, 84.1311798, -101.6346893, 84.3501587, -185.7369080, 185.7658539
19: -84.8784332, 47.2971611, -85.1241684, 47.4090195, -132.2874451, 132.4212952
20: -74.4952087, 57.2577209, -74.6858215, 57.3563728, -131.8515778, 131.9435425
21: -104.2269745, 62.7588425, -104.5306549, 62.9233475, -167.1503296, 167.2894897
22: -112.9534302, 72.6571960, -113.1113052, 72.7754745, -185.7289124, 185.7684937
23: -86.1931458, 58.0724030, -86.3863983, 58.2324295, -144.4255676, 144.4588013
24: -103.2813339, 69.0004730, -103.4666595, 69.1016693, -172.3829956, 172.4671326
25: -90.7378464, 67.8098602, -90.8747253, 67.9160919, -158.6539307, 158.6845856
26: -121.8225021, 89.1065063, -122.1417313, 89.3387909, -211.1612701, 211.2482300
27: -104.1350937, 73.7945557, -104.2617874, 73.8665543, -178.0016479, 178.0563354
28: -85.4373779, 62.9037514, -85.5646820, 62.9733047, -148.4106750, 148.4684296
29: -119.0569382, 76.2687759, -119.2156296, 76.4060287, -195.4629669, 195.4844055
30: -102.4688568, 79.1142349, -102.6765900, 79.3070068, -181.7758636, 181.7908325
31: -106.0280533, 66.6282196, -106.3238449, 66.8159332, -172.8439941, 172.9520569
32: -99.7366486, 73.2240829, -99.8550262, 73.3028412, -173.0394745, 173.0791016
33: -140.1805420, 80.4842682, -140.3839111, 80.6860962, -220.8666077, 220.8681641
34: -119.4638519, 72.6192093, -119.6387787, 72.7325668, -192.1964111, 192.2579651
35: -119.8588104, 70.1055298, -120.0825348, 70.2527008, -190.1115112, 190.1880493
36: -117.1950531, 69.5488815, -117.4026871, 69.6323929, -186.8274231, 186.9515686
37: -164.2515869, 73.7817612, -164.4193420, 73.8749466, -238.1265259, 238.2011108
38: -144.9993591, 86.0335159, -145.2219238, 86.2014236, -231.2007751, 231.2554321
39: -167.5591888, 77.7838821, -167.7944946, 77.9233475, -245.4825439, 245.5783691
40: -134.8826599, 73.6258087, -135.0688477, 73.7145386, -208.5971985, 208.6946411
41: -100.3878021, 66.9525070, -100.4963455, 67.0675125, -167.4553223, 167.4488525
42: -75.5094070, 65.0058746, -75.6105042, 65.2164764, -140.7258759, 140.6163635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
time: 120.06 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2308569
time: 101.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -124.3147278, 83.9410172, -124.8950958, 84.3449783, -208.6596985, 208.8360901
1: -69.7233429, 73.9899368, -70.1246033, 74.3148956, -144.0382233, 144.1145325
2: -62.4847717, 70.8895416, -62.9537354, 71.4621201, -133.9468689, 133.8432617
3: -71.8756409, 85.7175598, -72.4599457, 86.4771729, -158.3527832, 158.1775055
4: -75.0431824, 84.3226852, -75.5421677, 84.7985229, -159.8417053, 159.8648529
5: -67.1891632, 90.0692444, -67.7393875, 90.8504333, -158.0395966, 157.8086243
6: -102.3779373, 75.2557678, -102.7114868, 75.6792374, -178.0571747, 177.9672546
7: -83.0739670, 90.6673203, -83.6803055, 91.1276932, -174.2016602, 174.3476257
8: -88.2591782, 101.2195053, -88.7709656, 101.8145905, -190.0737610, 189.9904633
9: -77.9093628, 81.3977509, -78.2675400, 81.8044968, -159.7138519, 159.6652832
10: -110.3395386, 116.8626404, -111.1486282, 117.8286362, -228.1681824, 228.0112457
11: -110.3597946, 82.9094391, -111.2360687, 83.6199112, -193.9796753, 194.1455078
12: -110.2454605, 88.1911163, -111.3422623, 89.1543045, -199.3997498, 199.5333862
13: -109.4297333, 99.8250504, -110.0196838, 100.6015167, -210.0312500, 209.8447266
14: -161.9613037, 83.1143341, -162.9371643, 83.8653564, -245.8266296, 246.0514984
15: -90.8318329, 81.2744675, -91.4788589, 81.6712494, -172.5030670, 172.7533264
16: -117.7363892, 96.7426300, -118.2920914, 97.2989044, -215.0352783, 215.0347290
17: -163.6414032, 118.6682053, -164.6521912, 119.6038361, -283.2452393, 283.3203735
18: -101.2213364, 84.2074432, -101.9777222, 84.7541504, -185.9754944, 186.1851654
19: -84.8061905, 47.3621826, -85.4361725, 47.6503220, -132.4565125, 132.7983551
20: -74.3641357, 57.2756386, -74.8939514, 57.5571022, -131.9212341, 132.1695709
21: -104.1363373, 62.8400536, -104.9144592, 63.2851105, -167.4214478, 167.7545166
22: -112.4959106, 72.5006104, -113.0855103, 73.0741806, -185.5700836, 185.5861206
23: -86.1044006, 58.1120834, -86.6344910, 58.4682503, -144.5726318, 144.7465820
24: -103.1740799, 69.0260315, -103.6733398, 69.2966919, -172.4707489, 172.6993713
25: -90.5548630, 67.7298126, -90.9367294, 68.0867844, -158.6416321, 158.6665344
26: -121.1654816, 88.9492188, -122.3100433, 89.8221436, -210.9876251, 211.2592468
27: -103.9702835, 73.7841797, -104.5389328, 74.0854492, -178.0557251, 178.3231201
28: -85.3466492, 62.9038162, -85.8163452, 63.1569252, -148.5035706, 148.7201538
29: -118.7382889, 76.1751175, -119.3138275, 76.8077621, -195.5460510, 195.4889221
30: -102.3620911, 79.1129761, -102.9458618, 79.6311646, -181.9932556, 182.0588226
31: -105.9304352, 66.6733246, -106.6772079, 67.0604401, -172.9908752, 173.3505249
32: -99.5999222, 73.1700211, -99.9802017, 73.5048294, -173.1047516, 173.1502228
33: -140.1196289, 80.4029999, -140.6880188, 80.9043121, -221.0239105, 221.0910034
34: -119.3253555, 72.5018158, -119.8370056, 72.8480530, -192.1734009, 192.3388214
35: -119.7741394, 70.0023727, -120.3019104, 70.3562622, -190.1304016, 190.3042755
36: -117.1001053, 69.4883652, -117.5459747, 69.7144928, -186.8146057, 187.0343323
37: -164.0972290, 73.7511826, -164.6015625, 74.0391846, -238.1363678, 238.3527527
38: -144.9019775, 85.9428940, -145.5106812, 86.3069229, -231.2088928, 231.4535828
39: -167.4992065, 77.6972122, -168.0649414, 78.0973587, -245.5965576, 245.7621460
40: -134.7571411, 73.3863907, -135.2507019, 73.6786804, -208.4358215, 208.6370850
41: -100.2979279, 66.7581711, -100.6535034, 67.1187515, -167.4166718, 167.4116821
42: -75.4700851, 64.9775696, -75.7715836, 65.5173950, -140.9874878, 140.7491455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1995953
time: 122.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1410031
time: 99.79 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -124.5469818, 84.2570038, -124.9419174, 84.5080414, -209.0550232, 209.1989136
1: -69.8930664, 74.2049484, -70.1529236, 74.4239960, -144.3170624, 144.3578796
2: -62.6797905, 71.1357727, -62.9810333, 71.5947342, -134.2745056, 134.1168060
3: -72.1145554, 86.0621033, -72.4861298, 86.6595459, -158.7741089, 158.5482178
4: -75.2056732, 84.4283524, -75.5758209, 84.8445053, -160.0501709, 160.0041504
5: -67.4567413, 90.4546967, -67.7740936, 91.0550385, -158.5117798, 158.2287903
6: -102.5179901, 75.5680923, -102.7502213, 75.8294601, -178.3474426, 178.3183136
7: -83.4220428, 91.1353149, -83.7340240, 91.3830719, -174.8050842, 174.8693390
8: -88.4596024, 101.4829865, -88.7943039, 101.9517136, -190.4113159, 190.2772827
9: -78.1282959, 81.5927887, -78.3700256, 81.8358154, -159.9641113, 159.9627991
10: -110.6994629, 117.1922073, -111.3286057, 117.8683777, -228.5678253, 228.5208130
11: -110.4944229, 83.0338593, -111.3096619, 83.6124649, -194.1068878, 194.3435211
12: -110.8888550, 88.6424637, -111.6981201, 89.1935120, -200.0823669, 200.3405762
13: -109.7485657, 100.0821228, -110.1578217, 100.6718826, -210.4204407, 210.2399292
14: -162.5534363, 83.4390793, -163.2324219, 83.8899612, -246.4433899, 246.6714935
15: -91.2629089, 81.5084229, -91.6765518, 81.7174835, -172.9803925, 173.1849670
16: -117.9810257, 96.9664841, -118.3649216, 97.3708344, -215.3518372, 215.3314056
17: -164.0793457, 118.9882660, -164.8758698, 119.6354523, -283.7147827, 283.8641357
18: -101.4468231, 84.3203278, -102.0777130, 84.7808533, -186.2276611, 186.3980255
19: -84.9157104, 47.4134827, -85.4883270, 47.6638908, -132.5796051, 132.9018097
20: -74.5340881, 57.3544960, -74.9636383, 57.5768089, -132.1108704, 132.3181305
21: -104.2659302, 62.9406776, -104.9859314, 63.3051605, -167.5710907, 167.9265900
22: -112.9924774, 72.8032761, -113.3400879, 73.1150970, -186.1075745, 186.1433563
23: -86.2294235, 58.1861877, -86.6869965, 58.4904633, -144.7198792, 144.8731842
24: -103.3204193, 69.0979919, -103.7226181, 69.3229370, -172.6433563, 172.8206024
25: -90.7684860, 67.8982697, -91.0398254, 68.1281586, -158.8966370, 158.9380798
26: -121.8653717, 89.3414307, -122.6759186, 89.8560715, -211.7214050, 212.0173492
27: -104.1924362, 73.9211121, -104.5901718, 74.1467590, -178.3392029, 178.5112610
28: -85.4767761, 62.9942245, -85.8601837, 63.1925659, -148.6693115, 148.8544006
29: -119.0918579, 76.4772491, -119.4933167, 76.8440857, -195.9359131, 195.9705658
30: -102.5026321, 79.2823257, -102.9958954, 79.6951981, -182.1978302, 182.2781982
31: -106.0833740, 66.7448273, -106.7387238, 67.0764771, -173.1598358, 173.4835510
32: -99.7791748, 73.3200836, -100.0606079, 73.5418396, -173.3210144, 173.3806763
33: -140.3286743, 80.5241623, -140.7332001, 80.9612732, -221.2899475, 221.2573242
34: -119.5551300, 72.6664734, -119.8873062, 72.9256058, -192.4807434, 192.5537720
35: -119.9594421, 70.1327820, -120.3413086, 70.4181519, -190.3775940, 190.4740906
36: -117.2538757, 69.5817413, -117.5979309, 69.7596436, -187.0135193, 187.1796722
37: -164.3092651, 73.8449554, -164.6787109, 74.0713272, -238.3805847, 238.5236664
38: -145.1102448, 86.0629730, -145.5578613, 86.3605881, -231.4708252, 231.6208344
39: -167.6842041, 77.8139801, -168.1237183, 78.1488342, -245.8330078, 245.9376984
40: -134.9530640, 73.6151581, -135.2948303, 73.8013992, -208.7544403, 208.9099884
41: -100.4338760, 66.9835129, -100.6907959, 67.2262192, -167.6600952, 167.6742859
42: -75.5404892, 65.1395264, -75.8099823, 65.5509491, -141.0914307, 140.9495087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
time: 96.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2371882
time: 127.50 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -124.8633270, 84.1586151, -124.9355774, 84.2847061, -209.1480408, 209.0941620
1: -70.1144714, 74.1664124, -70.1781006, 74.2567902, -144.3712463, 144.3445129
2: -62.9766960, 71.1414566, -63.0183563, 71.2375488, -134.2142181, 134.1598053
3: -72.4983597, 86.0811157, -72.5654907, 86.2191772, -158.7175293, 158.6466064
4: -75.6477509, 84.5874023, -75.6352081, 84.6250153, -160.2727661, 160.2226105
5: -67.6801758, 90.4095078, -67.7860107, 90.5692444, -158.2494202, 158.1955261
6: -102.6295929, 75.6105423, -102.6917572, 75.7459946, -178.3755798, 178.3023071
7: -83.5183334, 90.8532028, -83.7076263, 91.0517426, -174.5700684, 174.5608215
8: -88.8063965, 101.5284271, -88.8546982, 101.6310425, -190.4374390, 190.3830872
9: -78.2027664, 81.6928558, -78.2451782, 81.8147278, -160.0174866, 159.9380341
10: -110.9208069, 117.9979172, -111.0455017, 118.1254425, -229.0462494, 229.0433960
11: -110.8905029, 83.9017639, -110.8996735, 83.8367081, -194.7272034, 194.8014374
12: -110.6960754, 89.1080246, -110.9567108, 89.3042221, -200.0003052, 200.0647278
13: -110.2088470, 100.2997513, -110.2422104, 100.4475403, -210.6563721, 210.5419464
14: -162.5291138, 83.9291382, -162.7601624, 84.0501633, -246.5792389, 246.6893005
15: -91.4535980, 81.4789734, -91.6128006, 81.6275635, -173.0811615, 173.0917664
16: -118.1451569, 97.4352036, -118.2398605, 97.5375214, -215.6826630, 215.6750641
17: -164.1531372, 119.8332672, -164.3127289, 119.8472519, -284.0003967, 284.1459961
18: -101.6905212, 85.0103607, -101.7609024, 84.9127655, -186.6032867, 186.7712708
19: -85.1833649, 47.7959366, -85.2059402, 47.7248878, -132.9082489, 133.0018768
20: -74.7009735, 57.6443291, -74.7652359, 57.6113205, -132.3122864, 132.4095612
21: -104.5838470, 63.4644127, -104.6057739, 63.3813095, -167.9651489, 168.0701904
22: -112.7788696, 72.9941254, -113.0137939, 73.1126099, -185.8914795, 186.0079041
23: -86.4206161, 58.6098366, -86.4578934, 58.5687256, -144.9893341, 145.0677338
24: -103.5141144, 69.4071808, -103.5744476, 69.3612518, -172.8753662, 172.9816284
25: -90.8146744, 68.1250458, -90.8981934, 68.1628189, -158.9774933, 159.0232239
26: -121.6559601, 89.7868195, -121.9533997, 89.9252625, -211.5812225, 211.7402039
27: -104.3117371, 74.1284637, -104.4288177, 74.0845947, -178.3963318, 178.5572815
28: -85.6036835, 63.1846237, -85.6546783, 63.1530151, -148.7566833, 148.8392944
29: -119.0248718, 76.7836075, -119.1751251, 76.8533936, -195.8782349, 195.9587402
30: -102.7158966, 79.7489777, -102.7709579, 79.7110596, -182.4269562, 182.5199280
31: -106.4242554, 67.2697144, -106.4564590, 67.2303391, -173.6545868, 173.7261658
32: -99.8892670, 73.4602661, -99.9504776, 73.4824371, -173.3717041, 173.4107361
33: -140.7695312, 80.7335663, -140.7991028, 80.7593536, -221.5288849, 221.5326538
34: -119.8240204, 72.7671890, -119.9227524, 72.8068085, -192.6307983, 192.6899414
35: -120.4251938, 70.2547455, -120.4795380, 70.2969360, -190.7221069, 190.7342834
36: -117.6498337, 69.6718826, -117.7060928, 69.6883926, -187.3382263, 187.3779755
37: -164.5059509, 74.0278168, -164.5952454, 74.0249176, -238.5308685, 238.6230469
38: -145.5494385, 86.2586517, -145.6085510, 86.2838669, -231.8333130, 231.8672028
39: -168.1605225, 77.9425201, -168.1874695, 77.9745255, -246.1350403, 246.1299744
40: -135.2640991, 73.5666122, -135.3507080, 73.6569672, -208.9210205, 208.9173279
41: -100.5874863, 67.0559692, -100.6415176, 67.1327209, -167.7202148, 167.6974792
42: -75.7104187, 65.5675812, -75.7081299, 65.5938416, -141.3042603, 141.2757111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=679, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1429404
time: 114.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1818057
time: 114.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -125.0885925, 84.4719849, -124.9820480, 84.4474716, -209.5360718, 209.4540253
1: -70.2793579, 74.3800049, -70.2060471, 74.3655396, -144.6448975, 144.5860596
2: -63.1670876, 71.3862457, -63.0455589, 71.3699493, -134.5370178, 134.4317932
3: -72.7331085, 86.4224091, -72.5917053, 86.4014511, -159.1345520, 159.0140991
4: -75.8040314, 84.6909866, -75.6684265, 84.6707306, -160.4747620, 160.3594055
5: -67.9441376, 90.7920380, -67.8209991, 90.7736053, -158.7177429, 158.6130219
6: -102.7656784, 75.9168472, -102.7305298, 75.8944168, -178.6600647, 178.6473694
7: -83.8612518, 91.3190689, -83.7612000, 91.3059998, -175.1672516, 175.0802612
8: -89.0014191, 101.7894363, -88.8777542, 101.7679291, -190.7693329, 190.6671906
9: -78.4199753, 81.8845215, -78.3519745, 81.8460083, -160.2659912, 160.2364807
10: -111.2778625, 118.3189774, -111.2253189, 118.1640854, -229.4419250, 229.5442963
11: -111.0079880, 84.0162430, -110.9723511, 83.8274994, -194.8354797, 194.9885864
12: -111.3365631, 89.5518112, -111.3123703, 89.3432159, -200.6797791, 200.8641815
13: -110.4970016, 100.5702286, -110.3820724, 100.5177994, -211.0148010, 210.9523010
14: -163.1169128, 84.2498093, -163.0547791, 84.0744629, -247.1913757, 247.3045959
15: -91.8588486, 81.7086945, -91.8038483, 81.6735382, -173.5323792, 173.5125427
16: -118.3761902, 97.6433868, -118.3125153, 97.6030121, -215.9792023, 215.9558716
17: -164.5886841, 120.1457672, -164.5361328, 119.8783569, -284.4670410, 284.6818848
18: -101.9100189, 85.1206284, -101.8585892, 84.9395752, -186.8495941, 186.9792175
19: -85.2910004, 47.8440018, -85.2580338, 47.7384109, -133.0294189, 133.1020203
20: -74.8686829, 57.7195587, -74.8347626, 57.6307793, -132.4994507, 132.5543060
21: -104.7102890, 63.5608406, -104.6771774, 63.4016037, -168.1118927, 168.2380066
22: -113.2734985, 73.2927170, -113.2671738, 73.1533508, -186.4268188, 186.5598907
23: -86.5403595, 58.6804123, -86.5104523, 58.5899811, -145.1303406, 145.1908569
24: -103.6520538, 69.4777985, -103.6231003, 69.3874512, -173.0394745, 173.1008911
25: -91.0254822, 68.2902985, -91.0006332, 68.2036667, -159.2291565, 159.2909241
26: -122.3508911, 90.1736374, -122.3187637, 89.9592056, -212.3100586, 212.4923706
27: -104.5257416, 74.2612610, -104.4785614, 74.1460037, -178.6717529, 178.7398224
28: -85.7320557, 63.2740669, -85.6981201, 63.1889763, -148.9210205, 148.9721680
29: -119.3743439, 77.0809097, -119.3535843, 76.8895721, -196.2639160, 196.4344940
30: -102.8494873, 79.9139862, -102.8205032, 79.7751160, -182.6246033, 182.7344971
31: -106.5623474, 67.3378754, -106.5173798, 67.2461014, -173.8084412, 173.8552551
32: -100.0649948, 73.6019897, -100.0300903, 73.5191193, -173.5841064, 173.6320801
33: -140.9741364, 80.8517914, -140.8440247, 80.8162460, -221.7903442, 221.6958160
34: -120.0509567, 72.9279785, -119.9728699, 72.8836365, -192.9346008, 192.9008484
35: -120.6062622, 70.3835907, -120.5190048, 70.3590927, -190.9653625, 190.9025879
36: -117.8004532, 69.7652817, -117.7580338, 69.7339706, -187.5344238, 187.5233154
37: -164.7135315, 74.1239548, -164.6717987, 74.0574265, -238.7709503, 238.7957458
38: -145.7531433, 86.3768158, -145.6558533, 86.3378143, -232.0909424, 232.0326538
39: -168.3400421, 78.0511627, -168.2456665, 78.0260391, -246.3660889, 246.2968292
40: -135.4548645, 73.7895813, -135.3938904, 73.7782516, -209.2330780, 209.1834564
41: -100.7199020, 67.2763672, -100.6783218, 67.2373276, -167.9572296, 167.9546814
42: -75.7771759, 65.7225800, -75.7462769, 65.6260376, -141.4032135, 141.4688568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=679, inp2_unstable=679, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1808117
time: 234.01 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
time: 290.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -124.6009750, 84.2927322, -124.9718552, 84.5535812, -209.1545563, 209.2645874
1: -69.9364243, 74.2602386, -70.1795273, 74.4538574, -144.3902893, 144.4397583
2: -62.6563454, 71.1908264, -62.9753456, 71.6261292, -134.2824707, 134.1661682
3: -72.1641235, 86.1595230, -72.5238037, 86.6930695, -158.8571930, 158.6833191
4: -75.2242737, 84.4899597, -75.5935516, 84.8726044, -160.0968628, 160.0835114
5: -67.4159393, 90.5257416, -67.7552261, 91.0852203, -158.5011597, 158.2809753
6: -102.5375900, 75.5255356, -102.7809677, 75.8107452, -178.3483276, 178.3064880
7: -83.3686600, 91.1657791, -83.7099457, 91.4094543, -174.7781067, 174.8757324
8: -88.4124603, 101.5390625, -88.7767715, 101.9929733, -190.4054260, 190.3158264
9: -78.2529144, 81.4355774, -78.4704819, 81.7492676, -160.0021667, 159.9060516
10: -110.8253860, 117.2509460, -111.3828125, 117.9118271, -228.7372131, 228.6337585
11: -110.7223053, 83.3793030, -111.3485107, 83.8316345, -194.5539093, 194.7278137
12: -110.9470444, 88.4654160, -111.7504272, 89.0982208, -200.0452576, 200.2158051
13: -110.2854614, 100.2551117, -110.4970169, 100.6957016, -210.9811707, 210.7521210
14: -162.7069244, 83.6755905, -163.2945862, 84.0463867, -246.7532959, 246.9701843
15: -91.3387985, 81.4900131, -91.7184982, 81.7358932, -173.0746918, 173.2085114
16: -118.0768814, 96.9884415, -118.4488907, 97.3911438, -215.4680176, 215.4373322
17: -164.2697144, 119.3225327, -164.9363251, 119.8593597, -284.1290588, 284.2588501
18: -101.6030045, 84.7310486, -102.1394272, 85.0401306, -186.6431274, 186.8704834
19: -85.0357742, 47.6322708, -85.5145874, 47.8038902, -132.8396606, 133.1468506
20: -74.6314697, 57.5248795, -74.9884491, 57.6820488, -132.3135071, 132.5133362
21: -104.4025269, 63.2202797, -105.0177307, 63.4819489, -167.8844604, 168.2380066
22: -113.0250473, 72.9673615, -113.3541260, 73.2275391, -186.2525940, 186.3214874
23: -86.3457031, 58.5038376, -86.7079315, 58.6837234, -145.0294189, 145.2117615
24: -103.4065170, 69.4775467, -103.7321777, 69.5557098, -172.9622192, 173.2097168
25: -90.8668823, 68.1022797, -91.0768051, 68.2643738, -159.1312561, 159.1790771
26: -122.0098724, 89.4925308, -122.7173767, 89.9545059, -211.9643555, 212.2098999
27: -104.1566238, 74.2856903, -104.5746078, 74.3731689, -178.5297852, 178.8602905
28: -85.5496445, 63.2145615, -85.8832321, 63.3292961, -148.8789368, 149.0977783
29: -119.1780930, 76.6962662, -119.5214844, 76.9866486, -196.1647186, 196.2177277
30: -102.6717377, 79.6743927, -103.0316315, 79.9346008, -182.6063232, 182.7060242
31: -106.2574615, 67.1047363, -106.7699966, 67.3031769, -173.5606232, 173.8747253
32: -99.8197174, 73.2668304, -100.1193008, 73.5111237, -173.3308411, 173.3861084
33: -140.5055542, 80.6062088, -140.8442993, 80.9782257, -221.4837646, 221.4504852
34: -119.6613007, 72.7456131, -119.9511185, 72.9724960, -192.6337891, 192.6967163
35: -120.1146774, 70.1769638, -120.4365692, 70.4433746, -190.5580444, 190.6135254
36: -117.4989548, 69.6202850, -117.7522507, 69.7781677, -187.2771301, 187.3725281
37: -164.4326477, 73.8716431, -164.7580261, 74.0863266, -238.5189667, 238.6296692
38: -145.3024902, 86.1975250, -145.6815796, 86.4045410, -231.7070312, 231.8791046
39: -168.0143738, 77.8772888, -168.3350525, 78.1608429, -246.1751709, 246.2123413
40: -135.0889282, 73.6221008, -135.3846893, 73.7975998, -208.8865356, 209.0067749
41: -100.5145721, 66.9913864, -100.7360077, 67.2256546, -167.7402039, 167.7273865
42: -75.5788651, 65.1687012, -75.8470459, 65.5666122, -141.1454773, 141.0157471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=679, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2243540
time: 104.88 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2389827, upper bound: 97.2817735
time: 97.50 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -125.2182770, 84.5180511, -125.3179321, 84.6376877, -209.8559570, 209.8359680
1: -70.3644104, 74.4101181, -70.4218750, 74.5146484, -144.8790588, 144.8320007
2: -63.2988281, 71.4140015, -63.3453217, 71.6823273, -134.9811554, 134.7593231
3: -72.8818741, 86.4594727, -72.9399185, 86.7862701, -159.6681519, 159.3993835
4: -75.9492340, 84.7211227, -76.0079346, 84.9529800, -160.9022217, 160.7290497
5: -68.0738068, 90.8272400, -68.1318054, 91.1703796, -159.2441864, 158.9590454
6: -102.8143158, 75.9370041, -102.9281006, 76.0364838, -178.8507996, 178.8651123
7: -83.9713135, 91.3524399, -84.0469284, 91.4803314, -175.4516296, 175.3993683
8: -89.1334915, 101.8237610, -89.1914673, 102.0731888, -191.2066650, 191.0152283
9: -78.5163116, 81.9202118, -78.5847778, 82.0180817, -160.5343933, 160.5049896
10: -111.3520355, 118.4836502, -111.5464783, 118.6275177, -229.9795380, 230.0301208
11: -111.0607910, 84.3363342, -111.4814453, 84.3943176, -195.4550934, 195.8177795
12: -111.3877258, 89.7588959, -111.8421326, 89.8477783, -201.2354736, 201.6010284
13: -110.6346512, 100.6408920, -110.6875076, 100.9057388, -211.5403900, 211.3283997
14: -163.2055511, 84.4293518, -163.4951172, 84.4799347, -247.6854858, 247.9244690
15: -91.8995361, 81.7549057, -92.0220642, 81.8739014, -173.7734070, 173.7769775
16: -118.4626846, 97.7186966, -118.6407166, 97.8049011, -216.2675629, 216.3594055
17: -164.6579590, 120.4352417, -165.0677032, 120.5018997, -285.1598511, 285.5029297
18: -101.9758377, 85.3165359, -102.3003464, 85.3738098, -187.3496399, 187.6168823
19: -85.3316040, 47.9652710, -85.6244583, 47.9969139, -133.3285065, 133.5897217
20: -74.9119110, 57.8219032, -75.1148834, 57.8540039, -132.7658997, 132.9367828
21: -104.7556152, 63.7484550, -105.1362152, 63.7877502, -168.5433350, 168.8846741
22: -113.3371124, 73.4533081, -113.5084991, 73.5037537, -186.8408661, 186.9617920
23: -86.5783234, 58.8156052, -86.8121948, 58.8602600, -145.4385834, 145.6278076
24: -103.6890488, 69.5866394, -103.8782883, 69.6140366, -173.3030853, 173.4649353
25: -91.0660248, 68.3912277, -91.1727753, 68.4231949, -159.4892273, 159.5639954
26: -122.4087982, 90.3973465, -122.8605042, 90.4722290, -212.8810272, 213.2578430
27: -104.5858002, 74.4103699, -104.8081665, 74.4378967, -179.0236969, 179.2185364
28: -85.7719116, 63.3756142, -85.9935608, 63.4145317, -149.1864471, 149.3691711
29: -119.4285965, 77.2933350, -119.6412659, 77.3322372, -196.7608337, 196.9346008
30: -102.8924942, 80.0980225, -103.1450195, 80.1711426, -183.0636292, 183.2430267
31: -106.6191864, 67.4944916, -106.9322357, 67.5281525, -174.1473389, 174.4267273
32: -100.1184311, 73.7012177, -100.2397003, 73.7602921, -173.8787231, 173.9409180
33: -141.1174927, 80.9003906, -141.1915588, 81.0955505, -222.2130432, 222.0919189
34: -120.1371307, 72.9940186, -120.2194595, 73.0881653, -193.2252808, 193.2134705
35: -120.7045670, 70.4205933, -120.7785797, 70.5306091, -191.2351685, 191.1991730
36: -117.8746719, 69.8061829, -117.9628220, 69.8678970, -187.7425690, 187.7690125
37: -164.7880402, 74.1809845, -164.9403992, 74.2477417, -239.0357819, 239.1213837
38: -145.8610992, 86.4147339, -145.9926147, 86.5026245, -232.3636780, 232.4073486
39: -168.4913483, 78.0839233, -168.5909576, 78.2532578, -246.7445984, 246.6748810
40: -135.5341492, 73.7763214, -135.6254883, 73.8674545, -209.4015961, 209.4017944
41: -100.7745361, 67.2996902, -100.8761978, 67.3928909, -168.1674194, 168.1758881
42: -75.8176575, 65.8778076, -75.9486389, 65.9727478, -141.7904053, 141.8264313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=502, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=680, inp2_unstable=680, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1327
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1311
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 647

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2888398
time: 114.19 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1910976, upper bound: 97.2888398
time: 124.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 240.95 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1410031
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1803784
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2308569
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1995953
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1410031
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2371882
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1429404
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1818057
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1808117
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2243540
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.2389827, upper bound: 97.2817735
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2888398
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 240.95
Output dim: 5, lower bound: -97.1910976, upper bound: 97.2888398

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -123.8367462, 83.8229523, -123.9281006, 83.9313889, -207.7681274, 207.7510376
1: -69.3940887, 73.9045258, -69.4714584, 74.0178833, -143.4119720, 143.3759766
2: -61.9761734, 70.8094482, -61.9956131, 70.9257050, -132.9018860, 132.8050537
3: -71.2956543, 85.5903168, -71.3686523, 85.7881927, -157.0838470, 156.9589691
4: -74.4732666, 84.2138062, -74.4600220, 84.2808685, -158.7541351, 158.6738281
5: -66.6662292, 89.9534607, -66.7472153, 90.1488037, -156.8150330, 156.7006836
6: -102.1862946, 74.9972000, -102.2305832, 75.0838165, -177.2701111, 177.2277832
7: -82.6274719, 90.5707245, -82.7809906, 90.7797546, -173.4072266, 173.3517151
8: -87.7020187, 101.1080627, -87.7163239, 101.2228470, -188.9248657, 188.8243866
9: -77.7162170, 81.0839081, -77.7741013, 81.1330719, -158.8492737, 158.8580017
10: -110.1101685, 115.9522324, -110.2953033, 116.0922318, -226.2023926, 226.2475281
11: -110.1813431, 82.0823975, -110.3837814, 82.1057739, -192.2870789, 192.4661713
12: -110.1151886, 87.1960297, -110.3720398, 87.3095245, -197.4247131, 197.5680695
13: -109.1393738, 99.5410767, -109.3900146, 99.8191605, -208.9585266, 208.9310913
14: -161.6753235, 82.4817963, -161.9927673, 82.6785278, -244.3538513, 244.4745636
15: -90.4791412, 81.0948868, -90.6486740, 81.2023926, -171.6815186, 171.7435455
16: -117.4701614, 96.3198395, -117.5728302, 96.3990936, -213.8692627, 213.8926544
17: -163.4450378, 117.7290573, -163.7267456, 117.8421478, -281.2871704, 281.4558105
18: -100.9936142, 83.6795731, -101.1571274, 83.7221069, -184.7157135, 184.8367004
19: -84.6540298, 47.0505409, -84.7691574, 47.0534515, -131.7074890, 131.8197021
20: -74.1942978, 57.0051231, -74.3270111, 57.0326195, -131.2269135, 131.3321228
21: -103.9731598, 62.3478851, -104.0976486, 62.3612900, -166.3344269, 166.4455261
22: -112.2911987, 72.0855789, -112.5243759, 72.2375336, -184.5287323, 184.6099548
23: -85.9585876, 57.8126411, -86.0930786, 57.8844223, -143.8430176, 143.9057159
24: -102.9759979, 68.8686600, -103.1184921, 68.9610443, -171.9370422, 171.9871521
25: -90.4241104, 67.4837952, -90.5647888, 67.5800476, -158.0041504, 158.0485840
26: -120.9714355, 88.1868286, -121.3759842, 88.3736115, -209.3450317, 209.5627899
27: -103.6731491, 73.5906219, -103.7700500, 73.6745758, -177.3477173, 177.3606415
28: -85.1937256, 62.7266846, -85.2932968, 62.7701683, -147.9638977, 148.0199890
29: -118.5789795, 75.6206436, -118.7779999, 75.7556458, -194.3346252, 194.3986511
30: -102.2102356, 78.6926575, -102.3973541, 78.7882233, -180.9984589, 181.0899963
31: -105.7034836, 66.3283539, -105.8893127, 66.3992844, -172.1027679, 172.2176514
32: -99.4333801, 72.8238907, -99.4663696, 72.8252258, -172.2585907, 172.2902527
33: -139.6180420, 80.2415314, -139.7091064, 80.3285904, -219.9466248, 219.9506378
34: -118.9612274, 72.3386536, -119.0986023, 72.3990326, -191.3602448, 191.4372559
35: -119.3269424, 69.8846970, -119.4376526, 69.9411163, -189.2680359, 189.3223572
36: -116.8267059, 69.3641434, -116.9635010, 69.3962097, -186.2229156, 186.3276367
37: -163.8520508, 73.5258026, -163.9748535, 73.5279236, -237.3799744, 237.5006561
38: -144.4763336, 85.8115845, -144.6010437, 85.9235535, -230.3998871, 230.4126282
39: -167.1095428, 77.5718689, -167.2361145, 77.6590576, -244.7686005, 244.8079834
40: -134.4435883, 73.3138733, -134.5671387, 73.4146118, -207.8581848, 207.8810120
41: -100.1094513, 66.5397186, -100.1912003, 66.6140747, -166.7235260, 166.7309265
42: -75.3324280, 64.4289703, -75.3250275, 64.4536209, -139.7860413, 139.7539978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1362438
time: 92.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1375865
time: 1521.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -124.1736450, 83.9069138, -124.5345001, 84.1543732, -208.3280182, 208.4414062
1: -69.6301575, 73.9652252, -69.8916168, 74.1650238, -143.7951660, 143.8568420
2: -62.3374252, 70.8649445, -62.6286697, 71.1475372, -133.4849548, 133.4936066
3: -71.7024384, 85.6836700, -72.0772171, 86.0872192, -157.7896423, 157.7608795
4: -74.8766556, 84.2935181, -75.1728592, 84.5095444, -159.3862000, 159.4663696
5: -67.0349960, 90.0384216, -67.3967361, 90.4494095, -157.4843750, 157.4351501
6: -102.3285828, 75.2108536, -102.5041046, 75.4835663, -177.8121490, 177.7149658
7: -82.9522552, 90.6415710, -83.3722763, 90.9530563, -173.9052734, 174.0138397
8: -88.1080322, 101.1876755, -88.4274292, 101.5057831, -189.6138153, 189.6151123
9: -77.8249588, 81.3472137, -78.0333252, 81.6113434, -159.4362946, 159.3805389
10: -110.2725220, 116.6511688, -110.8187027, 117.3065796, -227.5791016, 227.4698792
11: -110.3146210, 82.6363678, -110.7189178, 83.0546417, -193.3692322, 193.3552856
12: -110.2056961, 87.9321289, -110.8105469, 88.5884705, -198.7941589, 198.7426758
13: -109.3034668, 99.7520294, -109.6985016, 100.2010651, -209.5045319, 209.4505310
14: -161.8744049, 82.9079132, -162.4848633, 83.4244614, -245.2988434, 245.3927307
15: -90.7610550, 81.2288361, -91.1877289, 81.4621277, -172.2231750, 172.4165649
16: -117.6545792, 96.6996613, -117.9535141, 97.0854340, -214.7399597, 214.6531677
17: -163.5738525, 118.3567200, -164.1128235, 118.9380798, -282.5119324, 282.4695435
18: -101.1540222, 84.0044556, -101.5218048, 84.2981415, -185.4521484, 185.5262604
19: -84.7633286, 47.2378426, -85.0623016, 47.3805466, -132.1438751, 132.3001404
20: -74.3197479, 57.1716461, -74.6055527, 57.3246613, -131.6443939, 131.7771912
21: -104.0905304, 62.6470146, -104.4470291, 62.8826027, -166.9731293, 167.0940399
22: -112.4422760, 72.3381577, -112.8319397, 72.7105865, -185.1528625, 185.1700745
23: -86.0630112, 57.9874001, -86.3245544, 58.1916084, -144.2546234, 144.3119507
24: -103.1246948, 68.9243774, -103.3960724, 69.0679169, -172.1926117, 172.3204346
25: -90.5184860, 67.6322479, -90.7616501, 67.8611298, -158.3796082, 158.3938904
26: -121.1142197, 88.6938934, -121.7622833, 89.2663956, -210.3806152, 210.4561768
27: -103.9016342, 73.6519928, -104.1897888, 73.7958527, -177.6974792, 177.8417816
28: -85.3022156, 62.8067245, -85.5117340, 62.9255905, -148.2278137, 148.3184509
29: -118.6957092, 75.9524841, -119.0237274, 76.3422165, -195.0379028, 194.9762115
30: -102.3224182, 78.9240189, -102.6150970, 79.2040176, -181.5264282, 181.5391235
31: -105.8682327, 66.5468445, -106.2490005, 66.7821198, -172.6503448, 172.7958374
32: -99.5502777, 73.0641022, -99.7620468, 73.2476196, -172.7978821, 172.8261414
33: -139.9573059, 80.3574753, -140.3124084, 80.6195221, -220.5768280, 220.6698608
34: -119.2230759, 72.4487000, -119.5679321, 72.6445312, -191.8676147, 192.0166321
35: -119.6591721, 69.9707565, -120.0172195, 70.1828461, -189.8420105, 189.9879761
36: -117.0305557, 69.4510803, -117.3311615, 69.5785675, -186.6091309, 186.7822266
37: -164.0286255, 73.6725159, -164.3229523, 73.8187027, -237.8473206, 237.9954529
38: -144.7772675, 85.9081726, -145.1486053, 86.1381683, -230.9154053, 231.0567627
39: -167.3554688, 77.6633224, -167.7015076, 77.8637085, -245.2191772, 245.3648376
40: -134.6745605, 73.3831329, -135.0029907, 73.5652618, -208.2398071, 208.3861237
41: -100.2450562, 66.7091370, -100.4462433, 66.9260864, -167.1711426, 167.1553802
42: -75.4333344, 64.8276978, -75.5615082, 65.1541290, -140.5874634, 140.3892059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1760769
time: 129.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.2246222, upper bound: 97.1776435
time: 108.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -124.0683060, 84.1393356, -123.9773254, 84.0934982, -208.1618042, 208.1166687
1: -69.5643005, 74.1199341, -69.5023193, 74.1222763, -143.6865845, 143.6222534
2: -62.1699638, 71.0557938, -62.0248642, 71.0585251, -133.2284698, 133.0806427
3: -71.5321350, 85.9355087, -71.3962402, 85.9713211, -157.5034485, 157.3317566
4: -74.6341324, 84.3193665, -74.4954681, 84.3260422, -158.9601440, 158.8148346
5: -66.9320908, 90.3394775, -66.7832947, 90.3542633, -157.2863464, 157.1227722
6: -102.3249664, 75.3019943, -102.2697144, 75.2294312, -177.5543976, 177.5717010
7: -82.9719086, 91.0399475, -82.8371429, 91.0187073, -173.9906006, 173.8770905
8: -87.9011459, 101.3718262, -87.7415009, 101.3599243, -189.2610626, 189.1133270
9: -77.9393768, 81.2775116, -77.8762817, 81.1656799, -159.1050415, 159.1537933
10: -110.4701233, 116.2795563, -110.4761429, 116.1357269, -226.6058350, 226.7557068
11: -110.3107300, 82.2175980, -110.4574356, 82.1218872, -192.4326172, 192.6750336
12: -110.7586441, 87.6439590, -110.7276993, 87.3503265, -198.1089630, 198.3716125
13: -109.4810181, 99.7976151, -109.5580292, 99.8906784, -209.3717041, 209.3556519
14: -162.2679138, 82.8038406, -162.2835846, 82.7036438, -244.9715424, 245.0874329
15: -90.8999329, 81.3281403, -90.8448944, 81.2495270, -172.1494446, 172.1730347
16: -117.7114105, 96.5476379, -117.6481323, 96.4819565, -214.1933594, 214.1957703
17: -163.8828888, 118.0455627, -163.9508362, 117.8747025, -281.7575989, 281.9963684
18: -101.2160339, 83.7904205, -101.2551346, 83.7494202, -184.9654541, 185.0455627
19: -84.7639084, 47.1010780, -84.8213959, 47.0685043, -131.8323975, 131.9224701
20: -74.3640289, 57.0827789, -74.3958893, 57.0535622, -131.4175873, 131.4786682
21: -104.1027451, 62.4468765, -104.1687775, 62.3824348, -166.4851837, 166.6156464
22: -112.7905655, 72.3778000, -112.7820206, 72.2812653, -185.0718384, 185.1598206
23: -86.0832672, 57.8922043, -86.1454010, 57.9117851, -143.9950409, 144.0375977
24: -103.1206360, 68.9396973, -103.1688843, 68.9874420, -172.1080780, 172.1085815
25: -90.6384430, 67.6493759, -90.6691055, 67.6220245, -158.2604523, 158.3184814
26: -121.6738586, 88.5770111, -121.7285843, 88.4098892, -210.0837402, 210.3055725
27: -103.8910065, 73.7279358, -103.8223877, 73.7360992, -177.6271057, 177.5503235
28: -85.3230591, 62.8175316, -85.3373260, 62.8072090, -148.1302643, 148.1548615
29: -118.9326782, 75.9164810, -118.9583206, 75.7946243, -194.7272949, 194.8748016
30: -102.3498383, 78.8653183, -102.4479141, 78.8575287, -181.2073669, 181.3132019
31: -105.8527222, 66.4016190, -105.9509277, 66.4197998, -172.2725220, 172.3525391
32: -99.6122742, 72.9736481, -99.5485611, 72.8634644, -172.4757385, 172.5221863
33: -139.8242188, 80.3623962, -139.7543030, 80.3855438, -220.2097473, 220.1166992
34: -119.1882477, 72.5033340, -119.1486130, 72.4754486, -191.6636963, 191.6519470
35: -119.5104370, 70.0153809, -119.4780197, 70.0037842, -189.5142212, 189.4934082
36: -116.9791489, 69.4564438, -117.0154495, 69.4416046, -186.4207458, 186.4718933
37: -164.0645142, 73.6186066, -164.0530396, 73.5613556, -237.6258698, 237.6716461
38: -144.6832733, 85.9310226, -144.6498718, 85.9770126, -230.6602783, 230.5809021
39: -167.2939148, 77.6852036, -167.2962341, 77.7102203, -245.0041199, 244.9814148
40: -134.6370239, 73.5448151, -134.6123962, 73.5386658, -208.1756592, 208.1572113
41: -100.2438660, 66.7711487, -100.2285614, 66.7257385, -166.9696045, 166.9997101
42: -75.4009857, 64.5999908, -75.3618622, 64.4997482, -139.9007263, 139.9618530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1736401
time: 103.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1749454
time: 131.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -124.4047546, 84.2227020, -124.5802307, 84.3170776, -208.7218323, 208.8029327
1: -69.7984390, 74.1802521, -69.9186020, 74.2738495, -144.0722809, 144.0988464
2: -62.5312881, 71.1110840, -62.6551132, 71.2797470, -133.8110352, 133.7661896
3: -71.9404144, 86.0279694, -72.1026764, 86.2689819, -158.2093964, 158.1306458
4: -75.0375443, 84.3989105, -75.2047806, 84.5550232, -159.5925598, 159.6036987
5: -67.3016205, 90.4236755, -67.4309692, 90.6535950, -157.9551849, 157.8546448
6: -102.4677887, 75.5203094, -102.5419693, 75.6320572, -178.0998383, 178.0622864
7: -83.2986374, 91.1101227, -83.4249954, 91.2082748, -174.5069122, 174.5351105
8: -88.3069077, 101.4510193, -88.4495316, 101.6423721, -189.9492798, 189.9005432
9: -78.0481186, 81.5410767, -78.1423340, 81.6419983, -159.6901245, 159.6834106
10: -110.6320496, 116.9783478, -110.9975586, 117.3435593, -227.9755859, 227.9758911
11: -110.4461823, 82.7569046, -110.7904434, 83.0428696, -193.4890442, 193.5473328
12: -110.8488693, 88.3819885, -111.1658249, 88.6263962, -199.4752350, 199.5478210
13: -109.6322861, 100.0077972, -109.8435287, 100.2699509, -209.9022217, 209.8513184
14: -162.4663391, 83.2320175, -162.7788849, 83.4482574, -245.9145966, 246.0108948
15: -91.1910324, 81.4617767, -91.3837204, 81.5072174, -172.6982422, 172.8454742
16: -117.8973541, 96.9166565, -118.0247345, 97.1480408, -215.0453949, 214.9413910
17: -164.0115967, 118.6759033, -164.3356323, 118.9685440, -282.9801025, 283.0115051
18: -101.3774338, 84.1171875, -101.6180344, 84.3244629, -185.7019043, 185.7352295
19: -84.8725739, 47.2882156, -85.1136627, 47.3933792, -132.2659607, 132.4018707
20: -74.4889832, 57.2500725, -74.6746368, 57.3437386, -131.8327179, 131.9247131
21: -104.2196198, 62.7468338, -104.5174942, 62.9019508, -167.1215515, 167.2643280
22: -112.9383545, 72.6432724, -113.0847931, 72.7504425, -185.6887970, 185.7280579
23: -86.1875687, 58.0614662, -86.3764420, 58.2129517, -144.4005127, 144.4379120
24: -103.2703705, 68.9962234, -103.4445496, 69.0938492, -172.3642273, 172.4407654
25: -90.7319489, 67.8014374, -90.8642578, 67.9012756, -158.6332245, 158.6656799
26: -121.8143005, 89.0851822, -122.1271439, 89.2997589, -211.1140442, 211.2123108
27: -104.1224289, 73.7891388, -104.2391129, 73.8571014, -177.9795227, 178.0282593
28: -85.4318390, 62.8962440, -85.5547791, 62.9602661, -148.3920898, 148.4510040
29: -119.0497131, 76.2527695, -119.2026978, 76.3772278, -195.4269257, 195.4554749
30: -102.4620056, 79.0909271, -102.6644440, 79.2644653, -181.7264709, 181.7553711
31: -106.0195847, 66.6175766, -106.3086090, 66.7970428, -172.8166199, 172.9261780
32: -99.7293625, 73.2133484, -99.8418198, 73.2837677, -173.0131226, 173.0551758
33: -140.1657867, 80.4782486, -140.3568115, 80.6754761, -220.8412628, 220.8350525
34: -119.4522400, 72.6128006, -119.6175232, 72.7214050, -192.1736450, 192.2303162
35: -119.8441925, 70.1008606, -120.0559311, 70.2443085, -190.0885010, 190.1567841
36: -117.1838379, 69.5437775, -117.3823547, 69.6232529, -186.8070984, 186.9261322
37: -164.2400818, 73.7675323, -164.3987122, 73.8503265, -238.0903931, 238.1662292
38: -144.9840393, 86.0278931, -145.1947021, 86.1913300, -231.1753540, 231.2225647
39: -167.5400391, 77.7788315, -167.7594452, 77.9144669, -245.4544983, 245.5382690
40: -134.8700867, 73.6104126, -135.0458679, 73.6862869, -208.5563354, 208.6562805
41: -100.3802032, 66.9322433, -100.4825592, 67.0302124, -167.4104156, 167.4147949
42: -75.5030136, 64.9845581, -75.5989151, 65.1818314, -140.6848145, 140.5834503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=677, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.1732951, upper bound: 97.2265187
time: 149.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1749454
time: 99.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -123.9614334, 83.8535385, -124.2592621, 84.1162262, -208.0776520, 208.1127930
1: -69.4757996, 73.9264374, -69.6837082, 74.1631317, -143.6389313, 143.6101379
2: -62.1088905, 70.8308868, -62.2941322, 71.2354126, -133.3442993, 133.1250153
3: -71.4514313, 85.6188660, -71.7194977, 86.1696930, -157.6211243, 157.3383331
4: -74.6232071, 84.2379150, -74.7995148, 84.5617981, -159.1849976, 159.0374298
5: -66.8047409, 89.9793396, -67.0609894, 90.5418777, -157.3466187, 157.0403290
6: -102.2277451, 75.0119705, -102.4234467, 75.2237473, -177.4514923, 177.4354248
7: -82.7343674, 90.5929565, -83.0612793, 90.9492722, -173.6836395, 173.6542358
8: -87.8367538, 101.1357193, -88.0300674, 101.5248413, -189.3616028, 189.1657867
9: -77.7942123, 81.1241913, -77.9961700, 81.3077164, -159.1019287, 159.1203613
10: -110.1688919, 116.1341248, -110.6120911, 116.5614319, -226.7303162, 226.7461853
11: -110.2191086, 82.3331299, -110.8728790, 82.6323547, -192.8514404, 193.2060089
12: -110.1494904, 87.4272995, -110.8946991, 87.8245316, -197.9740295, 198.3219910
13: -109.2339706, 99.6043777, -109.6606674, 100.2017899, -209.4357452, 209.2650452
14: -161.7519531, 82.6719742, -162.4284210, 83.0894394, -244.8413849, 245.1003876
15: -90.5072708, 81.1330872, -90.8613434, 81.3990173, -171.9062805, 171.9944305
16: -117.5404739, 96.3350372, -117.8900452, 96.5709076, -214.1113586, 214.2250824
17: -163.5055237, 118.0159607, -164.2550049, 118.4632416, -281.9687500, 282.2709351
18: -101.0521698, 83.8678894, -101.6025085, 84.1515808, -185.2037506, 185.4703979
19: -84.6915207, 47.1663551, -85.1343079, 47.3079567, -131.9994659, 132.3006592
20: -74.2329025, 57.1013756, -74.6058350, 57.2527504, -131.4856567, 131.7072144
21: -104.0120773, 62.5292130, -104.5538635, 62.7426987, -166.7547760, 167.0830688
22: -112.3304443, 72.2301178, -112.7543640, 72.5761719, -184.9066162, 184.9844818
23: -85.9945908, 57.9260368, -86.3945847, 58.1417770, -144.1363678, 144.3206177
24: -103.0144348, 68.9660339, -103.3760376, 69.1821747, -172.1966095, 172.3420715
25: -90.4550171, 67.5714722, -90.7303162, 67.7914658, -158.2464905, 158.3017883
26: -121.0150452, 88.4207611, -121.9118423, 88.8898544, -209.9048920, 210.3326111
27: -103.7297821, 73.7173615, -104.1007843, 73.9545975, -177.6843719, 177.8181458
28: -85.2329407, 62.8169746, -85.5894012, 62.9888344, -148.2217712, 148.4063721
29: -118.6141739, 75.8280640, -119.0567780, 76.1925507, -194.8067322, 194.8848419
30: -102.2435760, 78.8602448, -102.7181320, 79.1746445, -181.4182129, 181.5783691
31: -105.7580872, 66.4443817, -106.3010025, 66.6593094, -172.4173889, 172.7453918
32: -99.4759979, 72.9189682, -99.6732407, 73.0636520, -172.5396423, 172.5922089
33: -139.7655334, 80.2815552, -140.0578308, 80.6051331, -220.3706512, 220.3393707
34: -119.0519257, 72.3850555, -119.3465958, 72.5897522, -191.6416626, 191.7316589
35: -119.4261169, 69.9122467, -119.6949310, 70.1068420, -189.5329590, 189.6071625
36: -116.8849792, 69.3967972, -117.1579971, 69.5221405, -186.4071045, 186.5547943
37: -163.9095764, 73.5895309, -164.2346802, 73.7243118, -237.6338501, 237.8241882
38: -144.5862122, 85.8408203, -144.9357300, 86.0821838, -230.6683960, 230.7765503
39: -167.2337952, 77.6014481, -167.5638733, 77.8857193, -245.1195068, 245.1653137
40: -134.5135956, 73.3028259, -134.7926331, 73.5007019, -208.0142822, 208.0954590
41: -100.1552582, 66.5717010, -100.3863831, 66.7743225, -166.9295807, 166.9580688
42: -75.3631592, 64.5602493, -75.5254135, 64.7861938, -140.1493530, 140.0856476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=501, inp2_unstable=501, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=678, inp2_unstable=678, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1327
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1311
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 538

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1932329
time: 109.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1375865
time: 1801.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 1913.84 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1362438
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1375865
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1760769
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.2246222, upper bound: 97.1776435
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1736401
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1749454
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.2265187
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1749454
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1932329
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1913.84
Output dim: 5, lower bound: -97.1732951, upper bound: 97.1375865
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1410031
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.2371882
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1429404
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1818057
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1808117
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2194675, upper bound: 97.1784137
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2243540
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.2389827, upper bound: 97.2817735
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.1621311, upper bound: 97.2888398
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1913.84
Output dim: 5, lower bound: -97.1910976, upper bound: 97.2888398

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 119.31 + 7995.23 = 8114.54 seconds
