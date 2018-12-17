data_path = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\data'
en_data = data_path + '\en.txt'
cn_data = data_path + '\cn.txt'
en_pre = data_path + '\en_pre.txt'
cn_pre = data_path + '\cn_pre.txt'

## Word segmentation
# with open(en_pre, 'w') as f_0:
#     with open(en_data, 'r') as en:
#         lines = en.readlines()
#         for i in lines:
#             for j in i.strip().split():
#                 f_0.write(j + ' ')
#             f_0.write('\n')

# with open(cn_pre, 'w', encoding='utf-8') as f_1:
#     with open(cn_data, 'r', encoding='utf-8') as cn:
#         lines = cn.readlines()
#         for i in lines:
#             temp = [j for j in i.split()]
#             real = []
#             for m in temp:
#                 real += [n for n in m]
#             for k in real:
#                 f_1.write(k + ' ')
#             f_1.write('\n')




