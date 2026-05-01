# Proyecto de Optimización con Temple Simulado
Este repositorio contiene el desarrollo completo del método de optimización basado en Temple Simulado, así como las herramientas utilizadas para evaluar su rendimiento y generar las visualizaciones incluidas en la memoria.

---

## Estructura del repositorio

- **`main.py`**  
  Ejecuta las 30 corridas del algoritmo, genera los resultados y produce todas las figuras utilizadas en la memoria.

- **`solution.py`, `vecinos.py`, `sa_timing.py`, `evaluate.py`, `robots.py`**  
  Módulos internos que implementan la lógica del algoritmo, la generación de vecinos, la evaluación del coste y los modelos cinemáticos.

- **`test/`**  
  Contiene ficheros auxiliares empleados para comprobar el funcionamiento de distintas partes del desarrollo.

- **`resultados_memoria/`**  
  Carpeta donde se almacenan las gráficas y resultados incluidos en la memoria final.

---

## Entorno de ejecución

Todo el proyecto ha sido ejecutado en un entorno **conda**, lo que garantiza la reproducibilidad y el aislamiento de dependencias. En cualquier caso, para instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```


La carpeta ``test/`` incluye algunos ficheros auxiliares utilizados para comprobar el funcionamiento de diferentes partes del desarrollo completo.

La carpeta ``resultados_memoria`` contiene las gráficas y resultados expuestos en la memoria.

Para ejecutar el algoritmo, ejecutar en la terminal ``python3 main.py``. En el propio directorio se generarán las gráficas y ficheros de resultados.