#coding: utf-8
# rmuh

import pandas as pd

#Agrupa las unidades económicas de acuerdo a los códigos de 3 dígitos del catálogo del SCIAN

#Unidades económicas:
u_names = ['111','112',
		'211','213',
		'236','237','238',
		'311','312','313','314','322','323','324','325','326','327','331','332','333','334','335','336',
		'431','432','433','434','436','435','437',
		'461','462','463','464','465','466','467','468','469',
		'481','482','484','485','486','487','488','492','493',
		'511','512','515','517','518',
		'521','522','523','524','525',
		'531','532','533',
		'541',
		'551',
		'561','562',
		'611',
		'621','622','623','624',
		'711','712','713','721','722',
		'811','812','813',
		'931','932']

#Archivo que contiene la lista de unidades económicas deshechadas
file = open('deleted.txt')
del_names = file.read().splitlines()

#Conteo de unidades económicas por ageb y por código en u_names
dfs = []
for i in range(1,33) :
	#print(i)
	#Se leen los archivos de unidades económicas por estado
	df = pd.read_csv('unidades_ageb/unidades_ageb_'+str(i)+'.csv', low_memory=False)
	#Se eliminan las unidades económicas con códigos en deleted.txt
	df = df.drop([x for x in df if str(x).startswith(tuple(del_names))], 1)
	#Se suman las unidades económicas por código en u_names
	for u_name in u_names :
		a_cols = df.columns.str.startswith(u_name)
		df = df.loc[:, ~a_cols].join(df.loc[:, a_cols].sum(1).rename(u_name))

	dfs.append(df)
	df.to_csv('runidades_ageb_'+str(i)+'.csv', encoding='utf-8-sig', na_rep=0, index=False)

data = pd.concat(dfs, sort=False)
data.to_csv('runidades_ageb.csv', encoding='utf-8-sig', na_rep=0, index=False)

#Se eliminan las ageb que no tengan unidades económicas de ningún tipo
#y se guarda el archivo con las clases dadas por el agrupamiento de k-means
df = pd.read_csv('runidades_ageb.csv', low_memory=False)
df = df.drop(['cluster2'], 1)
df1 = df.drop(['ENT','MUN','LOC','ageb','cluster1'], 1)
df.loc[:,'Total'] = df1.sum(axis=1)
df = df[ df['Total'] != 0]
df = df.drop(['Total'], 1)
df.insert(4, "cluster", df['cluster1'])
df = df.drop(['cluster1'], 1)
df.to_csv('runidades_kmeans3_ageb.csv', encoding='utf-8-sig', na_rep=0, index=False)

#Se eliminan las ageb que no tengan unidades económicas de ningún tipo
#y se guarda el archivo con las clases dadas por el agrupamiento de clara
df = pd.read_csv('runidades_ageb.csv', low_memory=False)
df = data.drop(['cluster1'], 1)
df1 = df.drop(['ENT','MUN','LOC','ageb','cluster2'], 1)
df.loc[:,'Total'] = df1.sum(axis=1)
df = df[ df['Total'] != 0]
df = df.drop(['Total'], 1)
df.insert(4, "cluster", df['cluster2'])
df = df.drop(['cluster2'], 1)
df.to_csv('runidades_clara3_ageb.csv', encoding='utf-8-sig', na_rep=0, index=False)

"""
#Se crea un archivo para observar la distribución de unidades económicas
#por ageb de X nivel socioeconómico y así analizar qué códigos de ue elimianr

df = pd.read_csv('runidades_kmeans3_ageb.csv', low_memory=False)
counts1 = df['cluster'].value_counts().to_dict()
#print("kmeans: ", counts1)
df = df.drop(['ENT','MUN','LOC','ageb'], 1)
df_new = df.groupby(df['cluster']).sum()
#print(df_new.info())

df_new.loc[:,'Total'] = df_new.sum(axis=1)

df_new.to_csv('runidades_kmeans3_ent.csv', encoding='utf-8-sig', index=True)
"""