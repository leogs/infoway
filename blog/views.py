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
import os

module_dir = os.path.dirname(__file__)  # get current directory
file_path = os.path.join(module_dir, 'static/files/stopwords.txt')

dataset = pandas.DataFrame()

stopwords = open(file_path).read()

def removeStopwords(doc):
    d = doc.split()
    res1  = [word for word in d if word not in stopwords]
    return ' '.join(res1)

def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/novo.html', {'posts': posts})

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, 'blog/post_detail.html', {'post': post})

def post_new(request):
    form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})

def upload_file(request):
    if request.method == 'POST':
        file1 = request.FILES['file']
        if file1:
            names = ['Bdata','Bprocedimento','Bcid','Bhospital','Bsexo','Bidade','Bqd_clin','Bresposta']
            dataset = pandas.read_csv(file1, skiprows=1, parse_dates=[0], names=names)
            del dataset['Bdata']
            description = dataset.describe().to_html()
            shape = dataset.shape
            head = dataset.head(20).to_html(classes='greyGridTable')
            #distribution = dataset.groupby('class').size()
            data_html = dataset.to_html(classes=["table table-striped table-hover"])

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

            dataset.to_csv(os.path.join(module_dir, 'static/files/clean.csv'))

            return render(request, 'blog/statistics.html', {'loaded_data': description, 'shape': shape, 'head':head})
            
    data = {}
    return render(request, 'blog/upload.html', data)
    #return render(request, 'blog/upload.html', {'form': form})

def upload_csv(request):
    data = {}
    if "GET" == request.method:
        return render(request, "blog/test.html", data)
    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            messages.error(request,'File is not CSV type')
            return HttpResponseRedirect(reverse("blog:upload_csv"))
        #if file is too large, return
        if csv_file.multiple_chunks():
            messages.error(request,"Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
            return HttpResponseRedirect(reverse("blog:upload_csv"))
 
        file_data = csv_file.read().decode("utf-8") 
        print(file_data)       
 
        
 
    except Exception as e:
        logging.getLogger("error_logger").error("Unable to upload file. "+repr(e))
        messages.error(request,"Unable to upload file. "+repr(e))
 
    return HttpResponseRedirect(reverse("blog:upload_csv"))