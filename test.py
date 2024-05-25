# def max_machines(T, test_cases):
#     results = []
#     for i in range(T):
#         R, C, D = test_cases[i][:3]
#         NR, NC, ND = test_cases[i][3:]
        
#         max_machines_ram = NR // R
#         max_machines_cpu = NC // C
#         max_machines_disk = ND // D
        
#         max = min(max_machines_ram, max_machines_cpu, max_machines_disk)
        
#         results.append(max)
#     return results
def sayed(machines, inp):
  res = []

  # Validate input length (should be even for paired machine-bank data)
  if len(inp) % 2 != 0:
    raise ValueError("Input length must be even (machine-bank pairs).")

  for i in range(0, len(inp), 2):
    needed = inp[i]  
    bank = inp[i + 1]  

    min_cycles = float('inf')  
    for j in range(len(needed)):
      cycles = bank[j] // needed[j] 
      min_cycles = min(min_cycles, cycles)  

    res.append(min_cycles) 

  return res

print(sayed(30,
    [[10, 3, 16], [7, 86, 45], [18, 23, 3], [87, 92, 49], [17, 9, 9], [87, 80, 83], [18, 15, 12], [61, 9, 93], [4, 3, 10], [101, 51, 73], [20, 8, 12], [46, 66, 53], [16, 2, 12], [74, 46, 35], [15, 21, 3], [14, 97, 2] , [6 ,12 ,2] ,  [71, 18, 96],
    [13, 18, 14],
    [60, 31, 3],
    [5, 16, 11],
    [54, 36, 8],
    [4, 18, 16],
    [49, 89, 62],
    [1, 4, 10],
    [23, 76, 52],
    [10, 20, 6],
    [75, 5, 82],
    [2, 16, 15],
    [13, 72, 50],
    [1, 20, 1],
    [22, 99, 15],
    [19, 6, 8],
    [51, 14, 31],
    [14, 18, 15],
    [93, 64, 25],
    [8, 6, 7],
    [44, 23, 45],
    [8, 15, 4],
    [17, 22, 56],
    [14, 18, 15],
    [94, 23, 37],
    [13, 2, 4],
    [41, 77, 35],
    [5, 19, 16],
    [41, 76, 90],
    [14, 10, 9],
    [8, 46, 92],
    [8, 17, 1],
    [18, 101, 29],
    [15, 22, 5],
    [36, 93, 17],
    [1, 12, 12],
    [47, 20, 1],
    [7, 1, 1],
    [90, 85, 60],
    [9, 5, 15],
    [44, 48, 96],
    [3, 11, 9],
    [9, 75, 59]]
))