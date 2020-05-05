def analysePicObj(A, B, DEBUG):
    import numpy as np
    from scipy.ndimage import rotate
    from scipy.optimize import curve_fit

    maxLowLimit=9
    startSigmaFrac=0.5
    sigmaDiffThreshold=0.01
    bkgWeight=0
    bkgImgWeight=1.05
    
    def gauss(x, a, b, c):
        return a*np.exp(-(x-b)**2/(2*c**2))
    
    jAnew=np.sum((B-A),0)
    jAmaxpos=np.argmax(jAnew)

    D=B-A
    newQ=D[:,jAmaxpos-5:jAmaxpos+5]
    newQ=np.sum(newQ,1)
    jBmax=np.max(newQ)
    fwhmA=0
    
    for x in range(0,len(newQ)):
        if newQ[x] > 0.5*jBmax:
            fwhmA=x
            break

    for x in range(len(newQ)-1,0,-1):
        if (newQ[x] > 0.5*jBmax):
            fwhmB=x
            break
        
    fwhm=fwhmB-fwhmA
    roiYstart=int(np.argmax(newQ)-1.5*0.5*fwhm)
    roiYend=int(np.argmax(newQ)+1.5*0.5*fwhm)

    C=B-bkgImgWeight*A

    # rotate image 1.75 deg as visually inspected
    #C = rotate(C, 1.75)
    C=C[roiYstart:roiYend,:]
    X=np.sum(C,0)

    bkgSub=(np.sum(X[0:10])+np.sum(X[-11:-1]))/20
    X=X-bkgSub
    X=X.clip(min=0)
    
    maxPos=np.argmax(X)
    maxRoiWidth=200
    
    startPos=int(maxPos-0.5*maxRoiWidth)
    endPos=int(maxPos+0.5*maxRoiWidth)
    origVectStart=startPos

    subX=X[startPos:endPos]
    xVect=np.arange(0,200)

    localMax=np.argmax(subX)
       
    if np.amax(subX) < maxLowLimit:
        return 0,0,0,0,0,0,0
    
    stepSize=3
    spotInt=1
    startPos=0
    endPos=0
    
    for winStop in range(localMax+stepSize,len(subX),stepSize):
        winStart=2*localMax-winStop
    
        #intensity
        tmpInt=np.sum(subX[winStart:winStop])
        if (tmpInt/spotInt-1) < 0.01:
            startPos=winStart
            endPos=winStop
            break

        spotInt=tmpInt
  
    probCent=maxPos
    probSigma=endPos-startPos
    
    workVect=subX[startPos:endPos]
    workVectX=xVect[startPos:endPos]
    totalIntensity=sum(workVect)
    avCenter=(workVect@workVectX)/totalIntensity
    avSigma=np.sqrt((workVect@((workVectX-avCenter)**2))/totalIntensity)
    
    startPosA=origVectStart+startPos
    endPos=origVectStart+endPos
    avCenter=origVectStart+avCenter

    return startPosA, endPos, avCenter, avSigma, totalIntensity, probCent, probSigma
