from 机器学习 import main
# from 机器学习调参 import main

names = ['all']
# names = ['all','ELCnet','PLHN','Tumorsen','MHL','ALMN','ALMN2']
best_acc,best_experiment=0,0

for name in names:
    open('HER2实验500.txt', 'a').write(name+'\n')
    for num in range(1,501):
        acc,experiment = main(num,name)
        if acc>best_acc:
            best_acc = acc
            best_experiment = num
        open('HER2实验500.txt', 'a').write(str(best_acc) + '   experiments: %d\n\n' % best_experiment)