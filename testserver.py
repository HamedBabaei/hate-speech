from roberta import Roberta

en = "RT #USER#: Funny how “15 days to slow the spread” turned into “maybe you can have a barbecue in July"
print("Result for English:\n", Roberta(lang='en').transform([en]))

es = "RT #USER#: Esperando para ver la captura de Jaime Perla. \n #USER# #USER# Que bueno Romeo con la ayud"
print("Result for Spanish:\n", Roberta(lang='es').transform([es]))

