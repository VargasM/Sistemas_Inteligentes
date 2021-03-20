import matplotlib.pyplot as plt

def normalize(x, stats):
    return (x - stats['mean']) / stats['std']

def trestbps_bin(row):
    
    if row['trestbps'] <= 80:
        value = 'Low'
    
    elif row['trestbps'] > 120:
        value = 'High'
    else:
        value = 'Normal'
        
    return value

def plotAxesSbubplot(data,group, parameter, xlabel, ylabel, leyend=[], kind='bar' ):
    ax = data.groupby(group)[parameter].value_counts(normalize=True).unstack(parameter).plot(kind=kind,figsize=(15,9),rot=0)
    if leyend:
        ax.legend([leyend[0],leyend[1]])
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel);
    plt.show()

    
'''
Function: trestbps_bin
    Function categorize blood preasure.
    This parameter could change a little with the age, however we will have a
    constant values for blood pressure bin edges as follows:
    Low < 80 < Normal < 120 < High
Parameter: 
    Data: DataSet
Return:
    Node
'''    
def trestbps_bin(row):
    
    if row['trestbps'] <= 80:
        value = 'Low'
    
    elif row['trestbps'] > 120:
        value = 'High'
    else:
        value = 'Normal'
        
    return value

    