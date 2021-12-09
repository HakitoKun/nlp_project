import tarfile
import urllib.request as libreq
import os, sys
import subprocess

def do_preprocessing(file):
    bashcommand = "pandoc test/"+file+" +RTS -M6000m -RTS --verbose --trace --mathjax -f latex -t plain --wrap=none -o test/test.txt"
    process = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return

def main(argv):
    doc = argv[0]
    print("doc  = ",doc)
    print(argv)
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
        do_preprocessing(file)

if __name__=='__main__':
    main(sys.argv[1:])
