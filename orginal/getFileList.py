def getFileList(dataPath, baseName):
    import fnmatch 
    import os
    import re
    
    def sort_nicely( l ):
        """ Sort the given list in the way that humans expect.
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        l.sort( key=alphanum_key )

    # get file list based on inparameters
    fileList=fnmatch.filter(os.listdir(dataPath), baseName)

    # make sure it's sorted in natural order
    sort_nicely(fileList)
    
    return fileList