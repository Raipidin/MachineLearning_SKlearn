from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#features 
# 1 bico
# 2 asa
# 3 faz cócó
galinha1 = [1, 1, 1]
galinha2 = [0, 1, 1]
galinha3 = [1, 0, 1]

passaro1 = [1, 1, 0]
passaro2 = [0, 1, 0]
passaro3 = [0, 1, 1]

# 0 - passaro
# 1 - galinha
train_x = [galinha1, galinha2, galinha3, passaro1, passaro2, passaro3]
train_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
model.fit(train_x, train_y)

animal_secreto = [1, 1, 1]

print(model.predict([animal_secreto]))

segredo1 = [1, 1, 1]
segredo2 = [1, 1, 0]
segredo3 = [0, 1, 1]

test_x = [segredo1, segredo2, segredo3]
test_y = [1, 0, 0]

previsoes = model.predict(test_x)

print(previsoes)

corretos = (previsoes == test_y).sum()

print(corretos) 

total = len(test_x)

taxa_de_acerto = corretos / total

#sem o accuraçy score
print('Taxa de acerto: %.2f' %(taxa_de_acerto * 100), '%')

#no accuraçy score o primeiro parametro é o valor real e o segundo é o valor previsto
accuracy = accuracy_score(test_y, previsoes)

print('Accuracy: %.2f' %(accuracy * 100), '%')