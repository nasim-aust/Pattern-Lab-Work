from Bio import SeqIO
from Bio import Seq

fileName = 'D:\Studies\\4.2\Pattern_Recognition\Assignments\Datasets\SRR6055422\SRR6055422_1.fastq'
line = int(input())
new_records = []
i = 1
id = ""
for record in SeqIO.parse(fileName, "fastq"):
    # id = record.id
    # n = id.split('.')

    if i == line:
        print(str(i) + ":" + record.seq)
    i += 1
    # print(record.description.split())
    if i == 1000:
        break
print("Total number of sequences: ")
# print(id.split('.')[1])
print(i-1)