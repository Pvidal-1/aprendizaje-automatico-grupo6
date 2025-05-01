# aprendizaje-automatico-grupo6

## 1.- 📂 Descripción y Estructura del Proyecto

El dataset utilizado en este proyecto es el siguiente:
[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)
Este es un conjunto de datos de pistas de Spotify de 125 géneros diferentes. Cada pista tiene características de audio asociadas. Los datos están en formato CSV

```bash
Deber_Semana2/
├── src/                  # Código fuente
├── dataset/              # Dataset original, descargado de https://www.kaggle.com/
├── imagenes/             # Imágenes generadas en en google Colab durante el desarrollo
└── README.md             # Este archivo
```
## 2.- 🔍 Análisis Exploratorio de Datos (EDA)
En esta sección, se presenta un resumen del análisis exploratorio realizado sobre el dataset. El objetivo principal fue obtener una comprensión inicial de los datos, visualizar las relaciones entre las variables e identificar variables irrelevantes.
### 2.1.- :pencil: Descripción de Variables y Clases
El dataset contiene las siguientes variables:

| Nº  | Variable           | Tipo de Dato | Descripción breve                                              |
|-----|--------------------|--------------|----------------------------------------------------------------|
| 0   | Unnamed: 0         | int64        | Índice automático del dataset                                  |
| 1   | track_id           | object       | ID único de la canción                                         |
| 2   | artists            | object       | Nombre(s) del/los artista(s)                                   |
| 3   | album_name         | object       | Nombre del álbum                                               |
| 4   | track_name         | object       | Título de la canción                                           |
| 5   | popularity         | int64        | Popularidad en plataforma                                      |
| 6   | duration_ms        | int64        | Duración en milisegundos                                       |
| 7   | explicit           | bool         | Contenido explícito (sí/no)                                    |
| 8   | danceability       | float64      | Nivel de facilidad para bailar                                 |
| 9   | energy             | float64      | Nivel de intensidad musical                                    |
| 10  | key                | int64        | Tono musical (clave)                                           |
| 11  | loudness           | float64      | Volumen medio en decibelios                                    |
| 12  | mode               | int64        | Modo: mayor o menor                                            |
| 13  | speechiness        | float64      | Presencia de voz hablada                                       |
| 14  | acousticness       | float64      | Grado de carácter acústico                                     |
| 15  | instrumentalness   | float64      | Probabilidad de ser instrumental                               |
| 16  | liveness           | float64      | Presencia de público en vivo                                   |
| 17  | valence            | float64      | Positividad o alegría percibida                                |
| 18  | tempo              | float64      | Velocidad (BPM) de la canción                                  |
| 19  | time_signature     | int64        | Compás de la canción                                           |
| 20  | track_genre        | object       | Género musical principal (Variable objetivo o Clase)           |

### 2.2 :pencil: Matriz de Correlación
![Gráfico de resultados](imagenes/matrizcorrelacion.png)

#### Análisis
- **energy** y **loudness**: Tienen la correlación más alta (0.76). Las canciones más energéticas tienden a ser más ruidosas.
- **acousticness** y **energy**: Tienen una correlación muy negativa (-0.73). Cuanto más acústica es una canción, menos energética suele ser.
- **danceability** y **valence**: Tienen una correlación de 0.48. Las canciones más bailables suelen ser más alegres.
- **popularity**: Tiene correlaciones muy bajas con casi todas las variables. La popularidad no parece tener relación con ninguna característica técnica de la canción.
- **key**: Tiene una correlación muy baja con casi todas las variables y prácticamente insignificante con el género.

#### Conclusiones de la Matriz de Correlación
1. Aunque hay variables como **popularity** y **time_signature** que tienen correlaciones muy bajas, pero podrían ser útiles combinadas con las otras variables.
2. La variable **key** tiene una relación insignificante con casi todas las variables, por lo que será eliminada del dataset.

#### Eliminación de Variables Irrelevantes
Se eliminaron las siguientes variables por no aportar valor al análisis o por generar ruido en los datos:

- `Unnamed: 0`: identificador de registro sin utilidad analítica.
- `track_id`: identificador alfanumérico único, no necesario para el modelo.
- `key`: tiene una relación insignificante con casi todas las variables
  
Al final quedaron 17 variables Y 1 CLASE O VARIABLE OBJETIVO (track_genre)

## 3.- ⚙️ Preprocesamiento
Dado que el dataset original contiene 114000 filas, se seleccionarán solo 5000 para agilizar la ejecución de las pruebas.
Se seleccionarán los primeros 5000 registros para limitar un poco la cantidad de clases a predecir (originalmente hay más de 100).
### 3.1 :pencil: Tratamiento de nulos, codificación de variables categóricas y escalado de variables numéricas.
```python
#ELIMINACION DE POSIBLES NULOS
df_exploratorio = df_exploratorio.dropna()
#SELECCIONO LOS PRIMEROS 5000 PARA LIMITAR UN POCO LAS CANTIDAD DE CLASES A PREDECIR (ORIGINALMENTE HAY MÁS DE 100)
df_aleatorio =  df_exploratorio[:5000] 
#SEPARAR VARIABLES DE CLASE O VARIABLE OBJETIVO X E Y
#A LA VARIABLE X LE QUITO LA COLUMNAS TRACK_GENRE QUE CORRESPONDE A LA CLASE O VARIABLE OBJETIVO QUE SE VA ANALIZAR
X = df_aleatorio.drop(columns=["track_genre"]) 
Y = df_aleatorio["track_genre"]
#CODIFICAR LAS VARIABLES CATEGORICAS EN X SI EXISTIERAN
X = pd.get_dummies(X)
#CODIFICAR LA VARIABLE OBJETIVO O CLASES
le = LabelEncoder()
Y_codificado = le.fit_transform(Y)
#ESCALADO DE VARIABLES O CARACTERISTICAS X
scaler = StandardScaler()
X_scalado = scaler.fit_transform(X)
```
### 3.2 :pencil: División en conjunto de entrenamiento y prueba (80/20).
```python
#test_size=0.2: El 20% de los datos serán usados para prueba.
#stratify=Y: garantiza que la proporción de clases (track_genre) sea igual en entrenamiento y prueba.
X_train, X_test, Y_train, Y_test = train_test_split(X_scalado, Y_codificado, test_size=0.2, random_state=42,stratify=Y_codificado)
```
- **Tamaño del subdataset aleatorio**: (5000, 18)
- **Tamaño del conjunto de entrenamiento**: (4000, 7655)
- **Tamaño del conjunto de prueba**: (1000, 7655)
## 3.- 🤖 Implementación de clasificadores
### 3.1.- 💻 Modelo 1: Árbol de decisión
-Se utilizaron los Siguiente hiperparámetros: DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, max_features= None, min_samples_leaf= 1, min_samples_split= 2, random_state=42)
