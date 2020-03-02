#coding:utf-8
from __future__ import division
import re
import csv
import sys
import urllib2
from subprocess import call
from bs4 import BeautifulSoup  

def getID_Name(uniprotID):
    page = urllib2.urlopen('https://www.uniprot.org/uniprot/' + uniprotID)   
    contents = page.read()   
    soup = BeautifulSoup(contents,"html.parser")
    gene = ''
    uniID = []
    uniName = []
    goID = []
    goName = []
    try:
        gene = str(soup.find('div', class_='entry-overview-content', id="content-gene").get_text())
    except:
        uniprotID = str(soup.find('div', class_='entryEmpty').find('a').get_text())
        page = urllib2.urlopen('https://www.uniprot.org/uniprot/' + uniprotID)   
        contents = page.read()   
        soup = BeautifulSoup(contents,"html.parser")
        gene = str(soup.find('div', class_='entry-overview-content', id="content-gene").get_text())
    for tag in soup.find('table', class_='databaseTable'):
        if 'Molecular function' in tag.get_text():
            uniKeywords = tag.findAll('a')
            for term in uniKeywords:
                match = re.search(r'<a href="/keywords/([\w\W]+)">([\w\W]+)</a>',str(term))
                uniID.append(match.group(1))
                uniName.append(match.group(2))

    goTerms = soup.find('ul', class_ = 'noNumbering molecular_function')
    if goTerms:
        for tag in goTerms:
            a = tag.find('a')
            goName.append(str(a.get_text()))
            goID.append(re.search(r'(GO:[\w\W]+?)"',str(a)).group(1))
    return gene, uniID, uniName, goID, goName

def main():
    # i = 0
    # uniID = set()
    # temp = ''
    # headers = { 'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' } 
    # goRe = r'<tr><td>Molecular function</td><td><span>(.+?)</span>'
    # keywordRe = r'<property type="term" value="F:(.+)"/>'
    with open('Uniprot_ID') as f, open('uniID_keywords.tsv','a') as w:
        # reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        # header = reader.next()
        # writer = csv.writer(csvout, dialect='excel-tab')
        # writer.writerow(header)
        header = 'Uniprot ID\tGene\tUniprotKB Keywords ID\tUniprotKB Keywords Name\tGO Keywords ID\tGO Keywords Name'
        # w.write(header + '\n')
        for line in f:
            if line.strip() == '':
                continue
            if ',' in line:
                uniID = line.split(',')[0].strip()
            else:
                uniID = line.strip()
            gene, keywordID, keywordName, goID, goName = getID_Name(uniID)
            if gene == -1:
                print(uni)
            uniprot_keywords_ID = '|'.join(term for term in keywordID)
            uniprot_keywords_name = '|'.join(term for term in keywordName)
            GO_keywords_ID = '|'.join(term for term in goID)
            GO_keywords_name = '|'.join(term for term in goName)
            w.write(line.strip() + '\t' + gene + '\t' + uniprot_keywords_ID + '\t' + uniprot_keywords_name\
             + '\t' + GO_keywords_ID + '\t' + GO_keywords_name + '\n')
        #     url = "https://www.uniprot.org/uniprot/" + uniID + '.xml'
        #     # req = urllib2.Request(url = "http://m.qiushibaike.com/hot/page/1",headers = headers)
        #     try:
        #         contents = urllib2.urlopen(url)
        #     except:
        #         print line
        #         contents = ''
        #         continue
        #     for row in contents:
        #         if re.search(keywordRe, row):
        #             keyword = re.search(keywordRe, row).group(1)
        #             temp += keyword + ','
        #     temp = temp.strip(',') + '\n'
        #     w.write(line + temp)
        #     temp = ''
                    


    # print count
    # print noCount
    # print total





if __name__ == '__main__':
    main()