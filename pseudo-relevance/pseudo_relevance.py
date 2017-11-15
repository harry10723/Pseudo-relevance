import operator
import time
from collections import defaultdict 
from collections import Counter
from math import log
from math import sqrt
from math import e
from numpy import random
DocumentDictionary = defaultdict(Counter)
RelevanceDocument = defaultdict(Counter)
DocumentIDF = defaultdict(int)
QueryIDF = defaultdict(int)
NewQueryIDF = defaultdict(int)
#DocumentSum = defaultdict(int)
QueryDictionary = defaultdict(Counter)
NewQueryDictionary = defaultdict(Counter)
BGLM = defaultdict(float)
PsmmAll = defaultdict(dict)
submission = defaultdict(list)
MaxQueryDictionary = defaultdict(int)
queryNumber = 0
documentNumber = 0
Alpha = 1.0
Betta = 1.0
MaxIteration = 30
MAPValue = 100
RelevanceDocumentNumber = 3


#read query_list.txt save at querylist
querylist = []
fq = open('query_list.txt','r') 
for line in fq:
    querylist.append(line.rstrip('\n'))
fq.close()
#read document_list.txt save at documentlist
documentlist = []
fd = open('doc_list.txt','r') 
for line in fd:
    documentlist.append(line.rstrip('\n'))
fd.close()
#read query
for qlist in querylist:
    result = []
    fq = open(qlist,'r')
    for line in fq:
        line = line.rstrip(' -1\n')
        result.extend(list(line.split(' ')))
    c = Counter(result)
    for term  in c:
        if(MaxQueryDictionary[term] < c[term]):
            MaxQueryDictionary[term] = c[term]
        QueryIDF[term] += 1
    QueryDictionary[qlist] = c
    queryNumber += 1
    fq.close()
#read document
for dlist in documentlist:
    fd = open(dlist,'r') 
    lines = fd.readlines()
    fd.close()
    l_list = lines[3:]
    result = []
    dict = defaultdict(int)
    for l in l_list:
        l = l.rstrip(' -1\n')
        result.extend(list(list(l.split(' '))))
    c = Counter(result)
    #DocumentSum[dlist] = sum(c.values())
    for term  in c:
        DocumentIDF[term] += 1
    DocumentDictionary[dlist] = c
    documentNumber += 1
    #if(documentNumber == 10):
    #    break

'''
fd = open('RelevanceDocument.txt','w')
fd.write('Query,RetrievedDocuments\n')
coun = 0

#找Relevance Document
for query in QueryDictionary :
    fd.write(query+',')
    c = QueryDictionary.get(query)
    rank = {}
    #find relevant documents
    for dict in DocumentDictionary:
        sigma_q = 0
        sigma_d = 0
        sum = 0
        #讀取一個term
        for term in c:
            wq = 0
            wd = 0
            #term對query的TF-IDF
            wq = (1 + QueryDictionary[query][term]) * log(queryNumber / QueryIDF[term])
            #wq = (0.5 + 0.5 * QueryDictionary[query][term] / MaxQueryDictionary[term]) * log(queryNumber / QueryIDF[term])
            #term對document的TF-IDF
            if(DocumentIDF[term] == 0):
                wd = 0
            else:
                wd = (DocumentDictionary[dict][term]) * log(documentNumber / DocumentIDF[term])

            number = wq * wd
            sigma_q = sigma_q + (wq * wq)
            sigma_d = sigma_d + (wd * wd)
            #term值加總
            sum = sum + number
        #把document對query的值存在dictionary裡
        if(sum != 0):
            number = sum / sqrt( sigma_q )
            number /= sqrt( sigma_d )
        rank[dict] = number
    #rank排序
    sorted_x = sorted(rank.items(), key=operator.itemgetter(1),reverse=True)
    count = 0
    for s in sorted_x:
        fd.write(s[0]+' ')
        count += 1
        if(count == MAPValue):
            break
    fd.write('\n')
    print(query+' done.')
#    coun += 1
#    if(coun == 2 ):
#        break
fd.close()
'''



#read BGLM.txt
fd = open('BGLM.txt','r') 
lines = fd.readlines()
fd.close()
for term in lines:
    termlist = []
    term = term.rstrip('\n')
    termlist.extend(list(list(term.split('   '))))
    f = float(termlist[1])
    f = e**f
    BGLM[termlist[0]] = f


#read RelevanceDocument.txt
fd = open('RelevanceDocument.txt','r') 
lines = fd.readlines()
lines = lines[1:]
fd.close()
for term in lines:
    doclist = []
    temp = []
    docu = []
    term = term.rstrip(' \n')
    temp = term.split(',')
    term = temp[1]
    c = Counter()
    doclist.extend(list(list(term.split(' '))))
    docu.append(doclist[0])
    docu.append(doclist[2])
    docu.append(doclist[4])
    #doclist = doclist [0:RelevanceDocumentNumber]
    for doc in docu:
        c += DocumentDictionary[doc]
    d = c + QueryDictionary[temp[0]]
    for t in d:
        NewQueryIDF[t] += 1
    RelevanceDocument[temp[0]] = c
    NewQueryDictionary[temp[0]] = d


# Query Modeling by Simple Mixture Model 
fd = open('result.txt','w')
for query in QueryDictionary :
    fd.write(query+',')
    c = RelevanceDocument.get(query)
    Psmm = defaultdict(float)
    PsmmW = defaultdict(float)
    rank = {}
    for Iteration in range(0 , MaxIteration):
        #E step
        for term in c :
            temp = 0
            if(Iteration == 0):
                Psmm[term] = random.random()
                temp = (1 - Alpha) * Psmm[term]
                PsmmW[term] = temp / ( temp + Alpha * BGLM[term] )
            else:
                temp = (1 - Alpha) * Psmm[term]
                PsmmW[term] = temp / ( temp + Alpha * BGLM[term] )
        #M step
        rd = RelevanceDocument[query]
        sum = 0
        for term in c :
            Psmm[term] += RelevanceDocument[query][term] * PsmmW[term]
            sum += Psmm[term]
        if(sum == 0):
            for term in c :
                Psmm[term] = 0
        else:
            for term in c :
                Psmm[term] /= sum
        print(query + ' tern ' + str(Iteration) + ' finish.')
    PsmmAll[query] = Psmm
    for term in Psmm:
        fd.write(term +'='+ str(Psmm[term])+' ')
    fd.write('\n')


start = time.time()
fd = open('NewRelevanceDocument.txt','w')
fd.write('Query,RetrievedDocuments\n')
#重新找Relevance Document
for query in QueryDictionary :
    fd.write(query+',')
    q = NewQueryDictionary.get(query)
    rank = {}
    #find relevant documents
    for dict in DocumentDictionary:
        sigma_q = 0
        sigma_d = 0
        sum = 0
        #讀取一個term
        for term in q:
            wq = 0
            wd = 0
            #term對query的TF-IDF
            #wq = (1 + Alpha * QueryDictionary[query][term] + Betta * RelevanceDocument[query][term] / RelevanceDocumentNumber) * log(1 + queryNumber / NewQueryIDF[term])
            wq = (1 + Alpha * QueryDictionary[query][term] + Betta * PsmmAll[query][term] ) * log(1 + queryNumber / NewQueryIDF[term])
            #term對document的TF-IDF
            if(DocumentIDF[term] == 0):
                wd = 0
            else:
                wd = (DocumentDictionary[dict][term]) * log(1 + documentNumber / DocumentIDF[term])
            number = wq * wd
            sigma_q = sigma_q + (wq * wq)
            sigma_d = sigma_d + (wd * wd)
            #term值加總
            sum = sum + number
        #把document對query的值存在dictionary裡
        if(sum != 0):
            number = sum / sqrt( sigma_q )
            number /= sqrt( sigma_d )
        rank[dict] = number
        
    #rank排序
    sorted_x = sorted(rank.items(), key = operator.itemgetter(1),reverse = True)
    count = 0
    for str in sorted_x:
        fd.write(str[0]+' ')
        count += 1
        if(count == MAPValue):
            break
    fd.write('\n')
    print(query+' done.')
fd.close()
end = time.time()

t = end - start
s = t % 60
t -= s
t /= 60
m = t % 60
t -= m
t /= 60
#print(str(t) + ' hr ' + str(m) + ' min ' + str(s) + ' s.')
'''
fd = open('NewRelevanceDocument.txt','r')
lines = fd.readlines()
fd.close()

fd = open('submission.txt','r')
fd.write('Query,RetrievedDocuments\n')
for line in lines:
    temp = []
    count = 0
    line = line.rstrip(' \n')
    temp = line.split(',')
    fd.write(temp[0]+',')
    line = temp[1]
    line.extend(list(list(l.split(' '))))
    for doc in line
        fd.write(doc+' ')
        if(count == MAPValue):
            break
fd.close()
'''