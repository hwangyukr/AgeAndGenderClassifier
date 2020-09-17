import numpy as np
import matplotlib.pyplot as plt
from lib.dataset import *
from lib.init import *
import seaborn as sns
import pandas as pd

labels = ['gender','age(10)','height(5cm)','weght(5kg)','waistcircumference','eye(L)','eye(R)',
'ear(L)','ear(R)','systole','relaxation','bloodsugar','cholesterol','triglyceride',
'HDL','LDL','hemoglobin','proteinuria','creatinine','AST','ALT',
'gamma','smoknig','drinking','tartar']

def get_every_data():
    trainset, testset = initDataset()
    x = np.concatenate((trainset.x.numpy(), testset.x.numpy()), axis=0)
    gender = np.concatenate((trainset.gender_t.numpy(), testset.gender_t.numpy()), axis=0)
    age = np.concatenate((trainset.age_t.numpy(), testset.age_t.numpy()), axis=0)

    total_data = np.concatenate(([gender.T], [age.T]), axis=0).transpose(1,0)
    total_data = np.concatenate((total_data, x), axis=1)
    print('Total Data : {0}'.format(total_data.shape))
    return total_data

def cut_off_outlier(vector1, vector2):
    len = vector1.shape[0]

    sort1 = np.sort(vector1)
    sort2 = np.sort(vector2)

    q1 = int(len/4)
    q3 = int(len/4) * 3

    d1 = (sort1[q3] - sort1[q1]) * 2
    d2 = (sort2[q3] - sort2[q1]) * 2

    ret2 = np.where(np.logical_and(\
        np.logical_and(vector1 >= sort1[q1]-d1, vector1 <= sort1[q3]+d1),\
        np.logical_and(vector2 >= sort2[q1]-d2, vector2 <= sort2[q3]+d2)))

    return vector1[ret2], vector2[ret2]

def calc_correlattion(data, i, j):
    feature1 = data[:,i]
    feature2 = data[:,j]
    #feature1, feature2 = cut_off_outlier(feature1, feature2)

    feature1 = feature1 - feature1.min()
    feature2 = feature2 - feature2.min()

    ret = np.absolute(np.corrcoef(feature1, feature2)[0][1])
    if ret==0 or ret!=ret:
        ret = 1e-7
    print('TEST', labels[i], labels[j], ret)
    return ret

def graph(data, i, j):
    feature1 = data[0:1000,i]
    feature2 = data[0:1000,j]
    if i == 0 or i == 1:
        sns.set()
        dict = {}
        dict[labels[i]] = feature1
        dict[labels[j]] = feature2
        df = pd.DataFrame(dict)
        print(labels[i], labels[j])
        sns.violinplot(x=labels[i], y=labels[j], data=df)
        plt.show()
        plt.close()
        return

    feature_coord = np.concatenate(([feature1.T], [feature2.T]), axis=0).transpose(1,0)

    sort_idx = np.argsort(feature1)
    plt.figure()
    plt.scatter(feature1, feature2, alpha=0.1)
    plt.title('{0} , {1}'.format(labels[i],labels[j]), fontsize=20)
    plt.show()
    plt.close(fig)


if __name__=='__main__':
    print('KaKao Internship Applicant : Lim Hwangyu')
    data = get_every_data()
    print(data[0])


    record_num = data.shape[0]
    tuple_num = data.shape[1]

    correlation = np.zeros((tuple_num, tuple_num))
    vx = []
    vy = []
    vc = []
    for i in range(tuple_num):
        #div = calc_correlattion(data, i, i)
        for j in range(tuple_num):
            corr = calc_correlattion(data, i, j) / 1
            correlation[i][j] = corr
            if i != j:
                vy.append(i)
                vx.append(j)
                vc.append(corr)
            #graph(data, i, j)

    vx = np.array(vx)
    vy = np.array(vy)
    vc = np.array(vc)
    idx = np.argsort(-vc)
    print(vy[idx][0:40])
    print(vx[idx][0:40])
    print(vc[idx][0:40])

    plt.figure()
    visualize_corr = correlation #(np.log(correlation) + correlation * 2) / 3
    visualize_corr = visualize_corr - visualize_corr.min()

    print(np.argmax(correlation))

    plt.imshow(visualize_corr, interpolation='none')
    plt.show()
    plt.close()
