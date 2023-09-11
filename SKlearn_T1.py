from sklearn.svm import LinearSVC

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
dados = [galinha1, galinha2, galinha3, passaro1, passaro2, passaro3]
classes = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
model.fit(dados, classes)

animal_secreto = [1, 1, 1]

model.predict([animal_secreto])