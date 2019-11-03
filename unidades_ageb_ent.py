#coding: utf-8
# rmuh

# Programa para crear un conjunto de datos de unidades económicas por ageb a partir del directorio denue.

import pandas as pd

dfs = [] #Lista que guarda los conjuntos datos por estado.

# Creación de los conjuntos de datos por estado
for i in range(1,33) :
#	print(i)

#	Se lee el archivo del directorio por estado con las columnas que se van a utilizar.
	df = pd.read_csv('denue_inegi_'+str(i)+'_.csv', usecols=['cve_ent', 'cve_mun', 'cve_loc', 'ageb', 'codigo_act'], 
					low_memory=False, dtype={'cve_ent' : 'str', 'cve_mun' : 'str', 'cve_loc' : 'str', 'ageb' : 'str'})

#	Se crea el identificador único conformado por la clave de entidad, clave de municipio, clave de localidad y clave de ageb.
	df['cve_geo'] = df['cve_ent'].astype(str) + df['cve_mun'].astype(str) + df['cve_loc'].astype(str) + df['ageb'].astype(str)

#	Se obtiene la tabla de frecuencias de tipos de actividades económicas por clave geográfica.
	df = pd.crosstab(df['cve_geo'], df['codigo_act'], dropna=False)

	df.to_csv('unidades_ageb/unidades_ageb_'+str(i)+'.csv', encoding='utf-8-sig', na_rep=0, index=True)
	df = pd.read_csv('unidades_ageb/unidades_ageb_'+str(i)+'.csv', low_memory=False, dtype={'cve_geo' : 'str'})

#	Se descompone la clave geográfica en las claves de entidad, municipio, localidad y ageb.
	df['col'] = df['cve_geo'].astype(str)
	df['col1'] = df['col'].str[:5].astype(str)
	df['col2'] = df['col'].str[-8:].astype(str)
	df.drop('col', axis=1, inplace=True)

	df.insert(0, "ent", df['col1'].str[:2])
	df.insert(1, "mun", df['col1'].str[-3:])
	df.insert(2, "loc", df['col2'].str[:4])
	df.insert(3, "ageb1", df['col2'].str[-4:]) 

	df.drop('col1', axis=1, inplace=True)
	df.drop('col2', axis=1, inplace=True)

	df.to_csv('unidades_ageb/unidades_ageb_'+str(i)+'.csv', encoding='utf-8-sig', na_rep=0, index=False)
	df = pd.read_csv('unidades_ageb/unidades_ageb_'+str(i)+'.csv', low_memory=False)

#	Se lee el archivo que contiene la clase a la que pertenece cada ageb.
	dfc = pd.read_csv('ageb_clustered_order_3.csv', low_memory=False)

# 	Se crea un nuevo conjunto de datos que contenga tanto las unidades económicas como la clase de las ageb del estado en turno.
	dfc = dfc[dfc['ENT'] == i]
	dfc = pd.merge(dfc, df, left_on=['ENT','MUN','LOC','ageb'], right_on=['ent','mun','loc','ageb1'], how='inner')
	dfc.drop(['ageb1','ent','mun','loc','cve_geo'], axis=1, inplace=True)

#	Se guarda el conjunto de datos actual
	dfs.append(dfc)
	dfc.to_csv('unidades_ageb/unidades_ageb_'+str(i)+'.csv', encoding='utf-8-sig', na_rep=0, index=False)

# Se crea un conjunto de datos que abarque todas las ageb de país.
data = pd.concat(dfs, sort=False)

#print(data.info())
data.to_csv('unidades_ageb/unidades_ageb.csv', encoding='utf-8-sig', na_rep=0, index=False)