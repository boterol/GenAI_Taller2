# GenAI_Taller2
Luis Fernando Botero


## Fase 1: Selección de Componentes Clave del Sistema RAG
Para la vectorización de los documentos de EcoMarket dentro del sistema RAG (Retrieval-Augmented Generation), se propone utilizar el modelo text-embedding-3-small de OpenAI. Este modelo produce representaciones vectoriales de 1536 dimensiones, más que suficientes para capturar las relaciones semánticas esenciales en los fragmentos de texto que manejará el sistema —principalmente descripciones de productos, políticas de servicio y respuestas frecuentes de atención al cliente.

Desde el punto de vista técnico, el modelo equilibra precisión y eficiencia computacional. Aunque su capacidad de representación es inferior a la del modelo large, su desempeño en tareas de búsqueda semántica, clasificación y recuperación de información es sobresaliente cuando se trabaja con chunks pequeños y consultas repetitivas, donde no se requiere una comprensión semántica profunda.

Su espacio vectorial multilingüe garantiza una interpretación precisa de textos en español, aspecto crucial para la atención al cliente de EcoMarket. En cuanto a costos, representa una opción altamente rentable: alrededor de USD 0.02 por millón de tokens procesados, lo que permite escalar el sistema sin aumentar significativamente el gasto operativo. Además, los embeddings se generan una sola vez por documento, por lo que el costo se mantiene controlado a largo plazo.

Desde la perspectiva de seguridad, OpenAI no utiliza los datos enviados para reentrenar sus modelos, asegurando la confidencialidad de la información procesada. Dado que EcoMarket no planea manejar datos personales o financieros en este flujo, el uso del servicio cloud es seguro y acorde con las buenas prácticas de protección de datos.

Como alternativa open-source, podría considerarse el modelo intfloat/multilingual-e5-large disponible en Hugging Face, que también ofrece un buen rendimiento multilingüe. No obstante, su implementación práctica requeriría igualmente el uso de un proveedor de modelos de lenguaje (LLM provider), como OpenAI u otro servicio equivalente, lo que en la práctica seguiría implicando que parte de la información sea procesada fuera del entorno local. En ese sentido, la solución propuesta no introduce riesgos adicionales y mantiene un balance adecuado entre eficiencia, costo y facilidad de integración.

---

Para la gestión de los embeddings en el sistema RAG de EcoMarket, se propone utilizar Qdrant como base de datos vectorial. Qdrant ofrece un equilibrio óptimo entre rendimiento, costo y flexibilidad de despliegue, aspectos cruciales para una empresa de comercio electrónico mediana en rápido crecimiento.

Su motor, desarrollado en Rust, garantiza búsquedas vectoriales de alta velocidad mediante índices HNSW, con excelente eficiencia incluso en grandes volúmenes de datos. A diferencia de servicios totalmente gestionados como Pinecone, Qdrant puede alojarse localmente o en contenedores Docker, eliminando costos de suscripción y permitiendo un control total sobre los datos, algo relevante para mantener la privacidad y reducir la dependencia de terceros.

Frente a competidores como Weaviate o ChromaDB, Qdrant se distingue por su simplicidad de integración y su soporte nativo para búsquedas híbridas (vector + filtros estructurados), que resultan muy útiles para combinar la similitud semántica con atributos propios de productos, como categoría o disponibilidad.



## Fase 2: Creación de la Base de Conocimiento de Documentos
En el caso de EcoMarket, los documentos más relevantes para el sistema RAG de atención al cliente serían tres:

* Un PDF con las políticas de devolución, que contiene información esencial sobre plazos, condiciones y procedimientos.

*  Un archivo CSV con el inventario de productos, útil para responder preguntas sobre disponibilidad o características.

* Un archivo JSON con preguntas frecuentes, que serviría como base para respuestas rápidas a consultas comunes.

Respecto al chunking, la segmentación debe equilibrar el contexto y la eficiencia computacional. Para el PDF de políticas de devoluciones, lo ideal sería usar fragmentos de alrededor de 700 tokens, lo bastante amplios para incluir informacion clave completa sin romper la coherencia semántica. En el caso del CSV, cada fila puede considerarse un chunk independiente, ya que cada producto representa una unidad semántica separada. Por último, para el JSON, lo más adecuado es generar un embedding por cada par pregunta–respuesta (key–value pair), ya que juntos conforman una unidad semántica completa que facilita una recuperación más precisa durante las consultas.

Mantener una dimensionalidad uniforme en los embeddings de todas las colecciones permite realizar operaciones cruzadas —por ejemplo, asociar una pregunta frecuente con un producto o con un punto de la política de devoluciones—, ampliando así las capacidades del sistema en casos especificos. Tambien es una opcion manejar distintas dimensiones (por ejemplo ampliando en el  caso del pdf y reduciendo en  el caso de el csv) pero complicaria la  implementacion a cambio de una ligera mejora en performance de retrieval para los casos de menor dimensionalidad. 

En conclusión, una estrategia de chunking moderadamente amplia (alrededor de 800 tokens para textos complejos como políticas, y más pequeños para datos estructurados), combinada con embeddings uniformes y ligeros, representa el mejor equilibrio entre eficiencia, costo y coherencia semántica para el sistema RAG de EcoMarket.


## Implementación
La implementación presentada corresponde a un prototipo del sistema RAG descrito en los textos anteriores, utilizando Ollama junto con el modelo all-mini-v6 para llevar a cabo la operación de los agentes. Este prototipo permite que los agentes respondan consultas de manera semántica, recuperando información relevante de los documentos de EcoMarket y generando respuestas coherentes a partir de los embeddings, sirviendo como prueba de concepto para la arquitectura y flujo de trabajo planteados.

### Ejecución
Aquí tienes una guía simple y clara que puedes pegar en el README de tu repo:

---

## Cómo ejecutar el proyecto

Sigue estos pasos para levantar y usar el proyecto:

1. **Clonar el repositorio** (si no lo has hecho aún):

2. **Instalar las dependencias de Python**:

```bash
pip install -r requirements.txt
```

3. **Crear el archivo de entorno `.env`**:

En la raíz del proyecto, crea un archivo llamado `.env` con el siguiente contenido:

```env
OPENAI_API_KEY=tu_api_key_de_openai_aquí
```

> Sustituye `tu_api_key_de_openai_aquí` por tu clave real de OpenAI.

4. **Levantar Qdrant en Docker**:

El proyecto utiliza Qdrant como base de vectores. Para levantarlo, ejecuta:

```bash
docker-compose up -d
```

Esto iniciará Qdrant en un contenedor de Docker.

5. **Ejecutar el programa**:

```bash
python main.py
```

6. **Usar el programa**:

* Elige el agente con el que quieras interactuar:
  1 - Devoluciones
  2 - Pedidos
  3 - Preguntas y Respuestas

* Sigue las instrucciones en pantalla para hacer consultas.

* Para regresar al menú de agentes, escribe `back`.

* Para salir del programa, escribe `exit`.

