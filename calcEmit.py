def calcEmit(dataVect, numPics, slitSep, convScreen, L, DEBUG): 
    import numpy as np

    # first check for any NaN, if exists set them to 0
    locCnt=0
    for myRow in dataVect:
        if np.isnan(myRow).any():
            dataVect[locCnt,:]=[0,0,0,0,0,0,0]
        locCnt=locCnt+1

    # Reshape based on num pic
    sigmaV=dataVect[:,3]
    sigmaV=sigmaV.reshape(int(len(sigmaV)/numPics),numPics)
    centerV=dataVect[:,2]
    centerV=centerV.reshape(int(len(centerV)/numPics),numPics)
    intV=dataVect[:,4]
    intV=intV.reshape(int(len(intV)/numPics),numPics)

    # Initz
    includeList=[]
    intVlist=[]
    centVlist=[]
    sigmaVlist=[]
    row=0

    # go through all intensitities, the idea is that 
    # we want at least half of the pictures for each slitSep
    # pos to be nonzero to use it
    for curV in intV:
        if np.count_nonzero(curV) < int(numPics/2)+1:
            row=row+1
            continue
        else:
            intVlist.append(np.sum(intV[row,:])/np.count_nonzero(curV))
            centVlist.append(np.sum(centerV[row,:])/np.count_nonzero(curV))
            sigmaVlist.append(np.sum(sigmaV[row,:])/np.count_nonzero(curV))

            includeList.append(row)
            row=row+1

    # make all lists into arrays
    includeList=np.asarray(includeList)
    intVlist=np.asarray(intVlist)
    centVlist=np.asarray(centVlist)
    sigmaVlist=np.asarray(sigmaVlist)


    # All data is now available, time to calculate emittance
    # All done straight from Zhangs paper 

    xsj=np.arange(-(len(includeList)/2)*slitSep+0.05,(len(includeList)/2)*slitSep-0.05,slitSep)*1e-3
    totInt=np.sum(intVlist)
    xmean=(intVlist@xsj)/totInt
    centIn=centVlist*convScreen
    xjprim=(centIn-xsj)/L
    xprim=(intVlist@xjprim)/totInt
    sigmaxj=sigmaVlist*convScreen/L
    eX=np.sqrt(((((intVlist@(xsj-xmean)**2)) * ( intVlist@sigmaxj**2 + intVlist@(xjprim-xprim)**2)) - (intVlist@(xsj*xjprim)-totInt*xmean*xprim)**2)/totInt**2)
    spotSize=np.sqrt((intVlist@(xsj-xmean)**2)/totInt)

    if DEBUG:
        print('Number of slits included / emittance unnorm / spot size')
        print(str(len(includeList)) + " / " + str(eX) + " / " + str(spotSize*1e3) )
        
    return len(includeList), eX, spotSize

