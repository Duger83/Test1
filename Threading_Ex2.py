import os
import multiprocessing

def get_number(root_file):
    target_file = open("path/"+root_file, "r")
    target_path = target_file.readline().replace('\\',"/")
    goal_file = open(target_path, "r")
    num = int(goal_file.readline())
    goal_file.close
    target_file.close
    return num
    
roots = [[f] for f in os.listdir("path") if f.endswith(".txt")]

if __name__ == "__main__":
    with multiprocessing.Pool(processes=10) as pool:
        list_num = pool.starmap(get_number, roots) 
        print("Итоговая сумма чисел =:", sum(list_num))