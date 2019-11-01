def calcF(precision, recall):
    if precision != 0.0 or recall != 0.0:
        return 2.0 * precision * recall / (precision + recall)
    else:
        return 0.0


n = int(input())
cnt = [list(map(int, input().split())) for i in range(n)]

selectedAndRelevant = [cnt[c][c] for c in range(n)]
selected =  [sum([cnt[i][c] for i in range(n)]) for c in range(n)]
relevant =  [sum(cnt[c]) for c in range(n)]
precision =  [(selectedAndRelevant[c] / selected[c] if selected[c] != 0 else 0.0) for c in range(n)]
recall =  [(selectedAndRelevant[c] / relevant[c] if relevant[c] != 0 else 0.0) for c in range(n)]
f = [calcF(precision[c], recall[c]) for c in range(n)]

total = sum([sum(row) for row in cnt])
weightedF = sum([f[c] * relevant[c] for c in range(n)]) / total
weightedPrecision = sum([precision[c] * relevant[c] for c in range(n)]) / total
weightedRecall = sum([recall[c] * relevant[c] for c in range(n)]) / total

print(calcF(weightedPrecision, weightedRecall))
print(weightedF)