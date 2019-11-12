import numpy as np

def sumAll(pGrid,r,c):
    if(r<0 or c<0 or r>25 or c>25):
        return -1
    else:
        sumh= np.sum(pGrid[r], axis=0)
        sumV=0
        for i in range(len(pGrid)):
            sumV+=pGrid[i][c]
        
        sumD=0
        for i in range(len(pGrid)):
            if(r-i)>=0 and (c-i)>=0:
                sumD+=pGrid[r-i][c-i]
            elif(r-i)>=0 and (c+i)<=25:
                sumD+=pGrid[r-i][c+i]
            elif(c-i)>=0 and (r+i)<=25:
                sumD+=pGrid[r-i][c-i]
            elif(c+i)<=25 and (r+i)<=25:
                sumD+=pGrid[r-i][c-i]
        
        return sumh+sumV+sumD


if __name__ == '__main__':
    
    pGrid = np.random.randint(5, size=(25, 25))
    print(pGrid)
    #print(np.sum(pGrid[0:][1], axis=1))
    print(sumAll(pGrid, 2 , 3))
    
