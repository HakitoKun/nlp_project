import tarfile
import urllib.request as libreq
import os, sys
import subprocess
from urllib.parse import urlparse

def do_preprocessing(file):
    bashcommand = "pandoc test/"+file+" +RTS -M6000m -RTS --verbose --toc --trace --mathjax -f latex -t plain --template=template.plain --wrap=none -o test/"+file[:-4]+".txt"
    process = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(error)
    if error is None:
        print("pas d erreurs")
    return

def process_url(pdf_url):
    url = urlparse(pdf_url)
    path = url.path
    elements = path.split("/")
    dl_url = "/".join([url.scheme+'://www.'+ url.netloc,'e-print',elements[-1][:-4]])
    return(dl_url)

def create_balise(file):
    pass

def main(argv):
    pdf_url = argv[0]
    doc = process_url(pdf_url)
    #with libreq.urlopen('https://arxiv.org/e-print/2112.04484') as url:
    with libreq.urlopen(doc) as url:
       r = url.read()
    #print(r)
    with open("test/test.tar", "wb") as f:
        f.write(r)
    tar = tarfile.open("test/test.tar")
    tar.extractall("test/")
    tar.close()
    abstract = ""
    text = ""
    tex_files = []
    for file in os.listdir("test"):
        if file.endswith(".tex"):
            tex_files.append(file)

    print(tex_files)
    for file in tex_files:
        create_balise(file)


    for file in tex_files:
        do_preprocessing(file)

if __name__=='__main__':
    main(sys.argv[1:])