import concurrent.futures as pool
import time

list_num = [
            [175, 790, 103, 479, 433, 406, 510, 377, 184, 777, 990, 321],
            [466, 538, 210, 881, 873, 411, 856, 965, 537, 243, 749, 229, 545,
            312, 651, 342, 328, 152, 473, 479, 257, 899, 362, 584, 888],
            [393, 488, 955, 771, 536, 922, 182, 564, 581, 998, 556, 813, 482, 940, 947, 267, 721, 944, 820, 993],
            [691, 532, 354, 831, 241, 724, 547, 580, 191, 742, 263, 149, 661, 804, 819, 246, 519, 698],
            [373, 235, 860, 591, 592, 227, 455, 113, 426, 181, 741, 723, 998, 667, 827],
            [924, 880, 704, 133, 538, 795, 364, 687, 775, 925, 445, 659, 883, 620, 391, 
            836, 779, 617, 528, 914, 424],
            [324, 643, 171, 208, 330, 306, 559, 927, 871, 284, 438, 644, 447, 893, 287],
            [993, 788, 192, 169, 549, 162, 324, 213, 277, 376, 391, 243, 749, 229, 545, 516, 260, 798],
            [497, 971, 765, 137, 543, 498, 583, 649, 558, 488, 882, 907, 589, 151, 724, 
            689, 134, 492, 124, 114, 147, 734, 524, 658, 441, 908, 192, 240],
            [575, 873, 922, 950, 375, 555, 351, 582, 659, 629, 619, 851, 661, 804, 819, 
            246, 519, 698, 286, 639, 593, 773, 157]
            ]

def process_number(number):
    time.sleep(0.2)
    return 2 * number
 
def process_list(my_list):
    list1 = [process_number(a) for a in my_list]
    return list1

with pool.ThreadPoolExecutor(max_workers=10) as executor: 
    futures = [executor.submit(process_list, x) for x in list_num]
    i = 0
    for future in pool.as_completed(futures):
        result = future.result()
        i = i + 1
        first_list_sum = sum(result)
        print(f"Сумма чисел в {i}-м обработанном списке: {first_list_sum}")