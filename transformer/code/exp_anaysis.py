import matplotlib.pyplot as plt
import codecs
import re


attention_path = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\result\cost'
transformer_path = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\transformer\results\loss'

a_f_mean_loss = []
with codecs.open(attention_path, 'r', 'utf-8') as a_f:
    line = a_f.readlines()
    num = len(line)
    for i in range(num):
        if 'iteration' in line[i]:
            try:
                a_f_mean_loss.append(float(re.findall(r"\d+\.?\d*", line[i-1])[1]))
            except:
                print('Miss!')
final_result = a_f_mean_loss[0]
a_f_mean_loss.pop(0)
a_f_mean_loss.append(final_result)

t_f_mean_loss = []
with codecs.open(transformer_path, 'r', 'utf-8') as t_f:
    line = t_f.readlines()
    num = len(line)
    for i in range(num):
        if 'After' in line[i]:
            try:
                t_f_mean_loss.append(float(re.findall(r"\d+\.?\d*", line[i])[1]))
            except:
                print('Miss!')

plt.plot(range(len(a_f_mean_loss)), a_f_mean_loss, label='Attention')
plt.plot(range(len(t_f_mean_loss)), t_f_mean_loss, label='Transformer')
plt.xlabel('Iteration')
plt.ylabel('Mean loss')
plt.legend()
plt.show()






