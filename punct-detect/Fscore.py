import sys
import re
import pandas as pd

def statistic(s):
    with open(s,'r',encoding="utf_8") as f:
        l = f.readlines()
    f = open(s+ '.stats' ,'w', encoding="utf_8")
    writer = pd.ExcelWriter(s+'.xlsx', engine='xlsxwriter')
    
    label = []
    for line in l:
        if line == '\n': continue
        # arr = filter(None,re.split('\s', line))
        arr = line.split(' ')
        
        tag = arr[-2]
        if tag != 'O' and tag not in label and len(tag) >= 1:
          label.append(tag)

    label.append('O')
    predict = []
    d = dict()
    inv_d = dict()
    i = 0
    for tag in label:
        d[tag] = i
        inv_d[i] = tag
        predict.append(i)
        i = i+1
    predict = predict[:-1]

    #punction=[".",",",";",":","!","?",u"\u2026"]
    num_class = len(label)
    cfm = [[0 for i in range(num_class)] for j in range(num_class)]
    correct = 0.0
    L = 0
    TP = 0.0
    FP = 0.0
    FN = 0.0

    #create confusion matrix
    for s in l:
        if s == '\n':
            continue
        L += 1
        s = s[:-1]
        s_temp = s.split(' ')
        # s_temp = filter(None, re.split('\s',s))
        true_idx = d[s_temp[-2]]
        predict_idx = d[s_temp[-1]]
        cfm[predict_idx][true_idx] += 1
        #accuracy:
        if true_idx == predict_idx:
            correct+= 1
            
        if true_idx == predict_idx and s_temp[-2] != 'O':
            TP += 1

        if true_idx in predict and predict_idx != true_idx:
            FN += 1
        if predict_idx in predict and predict_idx != true_idx:
            FP += 1    
        
    
    recall = []
    precision = []
    f_1 = []
    frequencies = []

    for i in range(num_class):
        tpI = float(cfm[i][i])
        denum_precision = sum(cfm[i])
        denum_recall = 0
        for j in range(num_class):
            denum_recall += cfm[j][i]

        try:
            r = float(tpI / (denum_recall))
        except ZeroDivisionError:
            r = 0

        try:
            p = float(tpI / (denum_precision ))
        except ZeroDivisionError:
            p = 0
             
        recall.append(r)
        precision.append(p)
        frequencies.append(denum_recall)
        try:
            f_score = 2*(r*p)/(r + p)
        except ZeroDivisionError:
            f_score = 0
        
        f_1.append(f_score)
    
    class_freq = 'CLASS FREQUENCY:\n '
    for j in range(num_class):
        class_freq += '\t' + inv_d[j] + ': ' + str(float(frequencies[j])*100 / L) + '%\n'
            ##############################################################
            #                  Macro average                             #
            ##############################################################
    macro_p= sum(precision)/len(precision)
    macro_r = sum(recall)/len(recall)
    macro_f1 = 2* (macro_p * macro_r) / (macro_p + macro_r)

    average = 'MACRO AVERAGE:\n\t Recall: ' + str(100*macro_r) + ' %\n\t Precision: ' + str(100*macro_p) + ' %\n\t F_1 score: '  + str(100*macro_f1) + ' %\n'   

            ##############################################################
            #                  Micro average                             #
            ##############################################################

    sumTP = 0.0
    micro_r = TP / (TP + FN)
    micro_p = TP / (TP + FP)
    micro_f = 2*(micro_r * micro_p) / (micro_r + micro_p)
    micro = 'MICRO AVERAGE:\n\t Recall: ' + str(100*micro_r) + ' %\n\t Precision: ' + str(100*micro_p) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'   


    ##################
    accuracy = correct / L
    line_break = '#'*50 + '\n'

    f.write(class_freq)
    f.write(line_break)
    f.write('OVERALL ACCURACY: ' +str(100*accuracy) + ' %\n')
    f.write(line_break)
    
    res = pd.DataFrame({' ': list(inv_d.values()) + ['Macro', 'Micro'],
                       'Recall': recall + [macro_r, micro_r],
                       'Precision':precision + [macro_p, micro_p],
                       'F_1 score': f_1 + [macro_f1, micro_f]}
                       )
    res.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    for i in range(num_class):
        f.write(inv_d[i] + ':\n')
        f.write('\t Recall: ' + str(100*recall[i]) + ' %\n')
        f.write('\t Precision: ' + str(100*precision[i]) + ' %\n')
        f.write('\t F_1 score: ' + str(100*f_1[i]) + ' %\n')
        f.write(line_break)

    f.write(average)
    f.write(line_break)
    f.write(micro)
    f.close()

if __name__ == "__main__":
    statistic('../result/test_result.txt')
    statistic('../result/train_result.txt')
