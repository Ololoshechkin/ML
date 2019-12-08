import operator
import math

def withSupport(x, calc):
    if abs(x) <= 1.0:
        return calc(x)
    else:
        return 0.0

def uniform(x):
    if abs(x) < 1.0:
        return 1
    else:
        return 0.0

def triangular(x):
    return withSupport(x, lambda x: 1 - abs(x))

def epanechnikov(x):
    return withSupport(x, lambda x: 0.75 * (1 - x ** 2))

def quartic(x):
    return withSupport(x, lambda x: 15.0/16.0 * ((1 - x ** 2) ** 2))

def triweight(x):
    return withSupport(x, lambda x: 35.0/32.0 * ((1 - x ** 2) ** 3))

def tricube(x):
    return withSupport(x, lambda x: 70.0/81.0 * ((1 - abs(x) ** 3) ** 3))

def gaussian(x):
    res = 1 / math.sqrt(2.0 * math.pi) * math.exp(-0.5 * (x ** 2))
    return res

def cosine(x):
    return withSupport(x, lambda x: math.pi / 4.0 * math.cos(math.pi / 2.0 * x))

def logistic(x):
  return 1 / (math.exp(x) + 2.0 + math.exp(-x))

def sigmoid(x):
    return (2.0 / math.pi) / (math.exp(x) + math.exp(-x))

def silverman(x):
    if abs(x) <= 25.0:
        return 0.5 * math.exp(-x / math.sqrt(2)) * math.sin(x / math.sqrt(2.0) + math.pi / 4.0)
    else:
        return 0.0

allKernels = {
    "logistic" : logistic, 
    "uniform" : uniform, 
    "triangular" : triangular,
    "epanechnikov" : epanechnikov,
    "quartic" : quartic, 
    "triweight" : triweight, 
    "tricube" : tricube, 
    "gaussian" : gaussian, 
    "cosine" : cosine, 
    "sigmoid" : sigmoid, 
    "silverman" : silverman,
    "id" : id
}

def cityblock(v1, v2):
    return sum([abs(v1[i] - v2[i]) for i in range(len(v1))])

def euclidean(v1, v2):
    res = math.sqrt(sum([((v1[i] - v2[i]) ** 2) for i in range(len(v1))]))
    return res

def chebyshev(v1, v2):
    return max([abs(v1[i] - v2[i]) for i in range(len(v1))])

allDistances = {
    "manhattan" : cityblock,
    "euclidean" : euclidean,
    "chebyshev" : chebyshev
}

def argmax(arr):
    n = len(arr)
    bi = 0
    for i in range(n):
        if arr[i] > arr[bi]:
            bi = i
    return bi

def kNN(newVector, objects, h, kernel, dist):
    res = 0.0
    nrm = 0.0
    for obj in objects:
        c = obj['class']
        v = obj['vector']
        d = dist(newVector, v)
        k = kernel(d / h) if h > 1e-9 else (1 if d < 1e-9 else 0)
        nrm += k
        res += c * k
    if nrm != 0:
        return res / nrm
    else:
        return sum([obj['class'] for obj in objects]) / len(objects)

def varKNN(newVector, objects, k, kernel, dist):
    objects2 = sorted(objects, key = lambda obj: dist(newVector, obj['vector']))
    distK = dist(newVector, objects2[k]['vector'])
    res = 0.0
    nrm = 0.0
    for obj in objects:
        c = obj['class']
        v = obj['vector']
        d = dist(newVector, v)
        k = kernel(d / distK) if distK > 1e-9 else (1 if d < 1e-9 else 0)
        nrm += k
        res += c * k
    if nrm != 0:
        return res / nrm
    else:
        return sum([obj['class'] for obj in objects]) / len(objects)

def solve():
    n, m = map(int, input().split())
    objects = []
    for i in range(n):
        l = list(map(int, input().split()))
        c = l[m]
        vec = l[:m]
        obj = {'class' : c, 'vector': vec}
        objects.append(obj)

    newVector = list(map(int, input().split()))
    dictName = input()
    dist = allDistances[dictName]
    kernelName = input()
    kernel = allKernels[kernelName]
    t = input()
    if t == 'fixed':
        h = int(input())
        print("%.15f"%kNN(newVector, objects, h, kernel, dist))
    else:
        k = int(input())
        print("%.15f"%varKNN(newVector, objects, k - 1, kernel, dist))

solve()