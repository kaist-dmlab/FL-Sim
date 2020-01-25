from matplotlib import pyplot as plt

def saveGraphOfLog(logFilePath, idxX, idxY, ymin=0, ymax=1, dpi=256):
    logFile = open(logFilePath, 'r')
    logFile.readline() # ignore the first title line
    
    xs = [] ; ys = []
    while True:
        line = logFile.readline()
        if not line: break
        tokens = line.rstrip('\n').split(',')
        xs.append(float(tokens[idxX]))
        ys.append(float(tokens[idxY]))
    xmin = min(xs) ; xmax = max(xs)
    
    graphFilePath = logFilePath[:-3] + 'png'
    fig = plt.figure()
    fig.set_size_inches(3, 3)
    plt.plot(xs, ys)
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])
    plt.savefig(graphFilePath, dpi=dpi)
    plt.close(fig)