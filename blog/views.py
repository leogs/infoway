from django.shortcuts import render
import logging
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .models import Post
from .forms import PostForm
from .forms import UploadFileForm
from django.shortcuts import render, get_object_or_404
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import os

module_dir = os.path.dirname(__file__)  # get current directory
file_path = os.path.join(module_dir, 'static/files/stopwords.txt')

dataset = pandas.DataFrame()

stopwords = open(file_path).read()

def removeStopwords(doc):
    d = doc.split()
    res1  = [word for word in d if word not in stopwords]
    return ' '.join(res1)

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    #from itertools import izip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/novo.html', {'posts': posts})

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, 'blog/post_detail.html', {'post': post})

def post_new(request):
    form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})

def train(request, alg):
    if request.method=='GET':
        a = alg
        if not a:
            return render(request, 'blog/invalido.html')
        else:
            lista = ['LR', 'LDA', 'KNN', 'CART', 'NB']
            if (a not in lista):
                return render(request, 'blog/invalido.html')
            else:
                file_path = os.path.join(module_dir, 'static/files/clean.csv')
                dataset = pandas.read_csv(file_path)
                array = dataset.values
                print(array)
                X = array[:,:-1]
                Y = array[:,-1]

                validation_size = 0.20
                seed = 7
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

                seed = 7
                scoring = 'accuracy'
                if (a == 'LR'):
                    model = LogisticRegression()
                if (a == 'LDA'):
                    model = LinearDiscriminantAnalysis()
                if (a == 'KNN'):
                    model = KNeighborsClassifier()
                if (a == 'CART'):
                    model = DecisionTreeClassifier()
                if (a == 'NB'):
                    model = GaussianNB()

                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                mean = cv_results.mean()
                std = cv_results.std()

                model.fit(X_train, Y_train)
                predictions = model.predict(X_validation)

                labels = ['Autorizada', 'Não autorizada']
                accuracy = accuracy_score(Y_validation, predictions)
                matrix = confusion_matrix(Y_validation, predictions)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(matrix)
                plt.title('Confusion matrix of the classifier')
                fig.colorbar(cax)
                ax.set_xticklabels([''] + labels)
                ax.set_yticklabels([''] + labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                file_path = os.path.join(module_dir, 'static/files/matrix.png')
                plt.savefig(file_path, dpi=200, format='png', bbox_inches='tight')
                plt.close()

                report = classification_report(Y_validation, predictions)
                plot_classification_report(report)
                file_path = os.path.join(module_dir, 'static/files/test_plot_classif_report.png')
                plt.savefig(file_path, dpi=200, format='png', bbox_inches='tight')
                plt.close()
                return render(request, 'blog/train.html', {'a': a, 'mean': mean, 'std': std, 'matrix': matrix, 'accuracy': accuracy, 'report': report})

def upload_file(request):
    if request.method == 'POST':
        file1 = request.FILES['file']
        if file1:
            names = ['Bdata','Bprocedimento','Bcid','Bhospital','Bsexo','Bidade','Bqd_clin','Bresposta']
            dataset = pandas.read_csv(file1, skiprows=1, parse_dates=[0], names=names)
            del dataset['Bdata']
            #description = dataset.describe().to_html(classes='greyGridTable')
            shape = dataset.shape
            headers = 'data | procedimento | cid | hospital | sexo | idade | qd_clin | resposta'
            #head = dataset.head(20).to_html(classes='greyGridTable')
            #distribution = dataset.groupby('class').size()
            #data_html = dataset.to_html(classes=["table table-striped table-hover"])

            dataset['Bprocedimento'] = dataset['Bprocedimento'].apply(lambda x: x.split(" ")[0])
            dataset['Bprocedimento'] = dataset['Bprocedimento'].apply(lambda x: int(x))
            dataset['Bcid'] = dataset['Bcid'].apply(lambda x: x.split(" ")[0] if (x != "Não informado.") else "Não informado")

            dataset['Bqd_clin'] = dataset['Bqd_clin'].apply(lambda x: x.lower())
            dataset['Bqd_clin'] = dataset['Bqd_clin'].apply(lambda x: removeStopwords(x))

            vectorizer = TfidfVectorizer(min_df=0.005)
            tfidf_result = vectorizer.fit_transform(dataset['Bqd_clin'])
            df1 = pandas.DataFrame(tfidf_result.toarray(), columns=vectorizer.get_feature_names())

            dataset.drop('Bqd_clin', axis=1, inplace=True)
            dataset = pandas.concat([dataset, df1], axis=1)

            dataset['Bresposta'] = dataset['Bresposta'].astype('category')
            dataset['Bresposta'] = dataset['Bresposta'].cat.codes

            dataset = pandas.get_dummies(dataset)

            dataset = dataset[[c for c in dataset if c not in ['Bresposta']] + ['Bresposta']]

            new_headers = ' | '.join(list(dataset.columns.values))

            dataset.to_csv(os.path.join(module_dir, 'static/files/clean.csv'))

            return render(request, 'blog/statistics.html', {'shape': shape, 'headers': headers, 'new_headers': new_headers})

    data = {}
    return render(request, 'blog/upload.html', data)
    #return render(request, 'blog/upload.html', {'form': form})