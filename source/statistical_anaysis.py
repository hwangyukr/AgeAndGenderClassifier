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

def calc_correlattion(data, i, j):
    feature1 = data[:,i]
    feature2 = data[:,j]
    feature1 = (feature1-feature1.min()) / (feature1.max()-feature1.min())
    feature2 = (feature2-feature2.min()) / (feature2.max()-feature2.min())
    return np.absolute(np.correlate(feature1, feature2))

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
    '''
    over_idx = np.where(data[:,18] >= 1.0)[0]
    under_idx = np.where(data[:,18] < 1.0)[0]
    print(len(over_idx) + len(under_idx))
    print(len(np.where(data[over_idx][:,0]<1)[0]) + len(np.where(data[under_idx][:,0]==1)[0]))
    exit(1)

    record_num = data.shape[0]
    tuple_num = data.shape[1]

    over_idx = np.where(data[:,19] >= 21.0)[0]
    under_idx = np.where(data[:,19] < 21.0)[0]
    print(len(over_idx) + len(under_idx))
    print(len(np.where(data[over_idx][:,0]<1)[0]) + len(np.where(data[under_idx][:,0]==1)[0]))
    exit(1)
    '''

    correlation = np.zeros((tuple_num, tuple_num))
    for i in range(tuple_num):
        div_factor = calc_correlattion(data, i, i)
        for j in range(i, tuple_num):
            corr = calc_correlattion(data, i, j)
            correlation[i][j] = corr / div_factor
            correlation[j][i] = corr / div_factor
            graph(data, i, j)

    plt.figure()
    visualize_corr = np.log(correlation)
    visualize_corr = visualize_corr - visualize_corr.min()

    plt.imshow(-np.log(visualize_corr), interpolation='none')
    plt.show()
    plt.close()
