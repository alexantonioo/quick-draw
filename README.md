# Adivina Adivinador

**Adivina Adivinador** es una aplicación web 100 % offline que reconoce dibujos y muestra las **3 predicciones más probables** en tiempo real. Se puede usar de dos formas:

- **Modo Ratón:** dibujas directamente sobre el lienzo de la pantalla.
- **Modo Pizarrón:** una webcam enfoca una pizarra blanca real y la app interpreta lo que dibujas con rotulador.

Todo el procesamiento ocurre en el navegador con **TensorFlow.js**. No se envía nada a internet ni se necesita conexión durante la feria.

---

## Autores

Proyecto desarrollado por:

- **Alex Contreras**
- **Jasneidy Pernia**

Para el **XXVI Encuentro con la Física, Química, Matemática y Biología**.

---

## Características

- Reconocimiento de **24 categorías** con un modelo TensorFlow.js local (`graph-model`).
- Predicciones **Top 3** con porcentaje de confianza.
- Visualizador de lo que "ve" la IA: tensor de **28 × 28 px** en escala de grises.
- Lectura en voz alta del resultado (se puede silenciar).
- Preprocesamiento con recorte automático del trazo, centrado y normalización.
- Calibración guiada de la pizarra: zona vigilada + captura de la pizarra vacía.
- Selector de cámara para elegir entre la integrada y una webcam externa.
- Interfaz tipo plano técnico (blueprint), pensada para verse desde lejos en una feria.
- **Sin dependencias externas en tiempo de ejecución:** TensorFlow.js, fuentes y modelo están incluidos en el proyecto.

---

## Requisitos

- Un navegador moderno: **Google Chrome** o **Microsoft Edge** (recomendados).
- Un servidor local para abrir la app. **No funciona con doble clic sobre `index.html`.**
- Opcional: una webcam para el modo pizarrón.

> **¿Por qué no sirve abrir `index.html` directamente?**
> Con el protocolo `file://` el navegador bloquea la carga del modelo por seguridad (CORS) y la cámara exige un contexto seguro. Por eso hay que servir la carpeta por `http://localhost` (o `http://127.0.0.1`).

---

## Estructura del proyecto

```text
quick-draw/
├── index.html                  # Aplicación principal
├── styles.css                  # Estilos de la interfaz
├── app.js                      # Lógica: dibujo, cámara, modelo, voz
├── tf.min.js                   # TensorFlow.js 4.22.0 (local)
├── model/
│   ├── model.json              # Arquitectura del modelo (graph-model)
│   ├── group1-shard1of1.bin    # Pesos del modelo
│   └── class_names.json        # Nombres y orden de las 24 clases
├── fonts/                      # Tipografías locales
├── tools/
│   ├── prueba-camara.html      # Página para afinar umbrales de la cámara
│   └── verificar-modelo.js     # Validador del modelo sin navegador
├── INICIAR.bat                 # Lanzador del servidor portátil (Windows)
├── .gitignore
└── README.md
```

---

## Paso a paso para probar la web

### 1. Copiar el proyecto

Descarga o clona el repositorio completo:

```bash
git clone https://github.com/alexantonioo/quick-draw.git
```

También puedes descargar el ZIP desde GitHub y descomprimirlo. **No muevas archivos fuera de sus carpetas**: las rutas son relativas.

> **Para la feria:** prepara la carpeta en tu equipo (con `caddy.exe` incluido), cópiala a un USB y luego al disco de la laptop. Evita ejecutarla directamente desde el USB, que suele ser más lento.

### 2. Verificar que estén los modelos

La carpeta `model/` debe contener exactamente estos tres archivos:

```text
model/
├── model.json
├── group1-shard1of1.bin
└── class_names.json
```

- Si clonaste el repositorio, ya vienen incluidos.
- Si te pasaron el modelo aparte (USB, correo, etc.), copia los tres archivos dentro de `model/`.
- El nombre del `.bin` debe coincidir con el que aparece dentro de `model.json`. Si tu archivo se llama distinto, renómbralo a `group1-shard1of1.bin`.

Puedes comprobar que el modelo está bien con Node.js instalado (solo en el equipo de desarrollo):

```bash
node tools/verificar-modelo.js
```

Si todo está correcto verás un mensaje similar a:

```text
=== TODO CORRECTO. Listo para la feria ===
```

### 3. Servir la carpeta por HTTP

Elige **una** de estas opciones.

#### Opción A — Servidor portátil (recomendado, sin instalar nada)

Ideal para la laptop de la feria, aunque no tenga Python ni Node.

1. Descarga **Caddy** para Windows (amd64) desde <https://caddyserver.com/download>.
2. Descomprime el ZIP.
3. Copia `caddy.exe` dentro de la carpeta del proyecto (junto a `INICIAR.bat`).
4. Haz doble clic en **`INICIAR.bat`**.
5. Se abrirá automáticamente `http://localhost:8000` en el navegador.
6. Para detener el servidor, cierra la ventana llamada **"Servidor Adivina Adivinador"**.

> Si Windows SmartScreen muestra un aviso por el `.exe` descargado, pulsa **Más información → Ejecutar de todas formas**.

#### Opción B — Con Python instalado

Desde la carpeta del proyecto:

```bash
python -m http.server 8000
```

Luego abre <http://localhost:8000>.

En algunos sistemas el comando es `python3`:

```bash
python3 -m http.server 8000
```

#### Opción C — Con Visual Studio Code

1. Abre la carpeta del proyecto en VS Code.
2. Instala la extensión **Live Server**.
3. Haz clic derecho sobre `index.html` → **Open with Live Server**.

### 4. Abrir la aplicación

Abre en el navegador:

```text
http://localhost:8000
```

**Importante:** usa `localhost` o `127.0.0.1`. No uses `http://192.168.x.x` (la cámara no funciona por IP local sin HTTPS) ni abras el archivo con doble clic.

### 5. Probar el modo Ratón

1. Espera a que el indicador del modelo pase a **LISTO**.
2. Dibuja sobre el lienzo con el ratón.
3. Verás el Top 3 de predicciones y escucharás el resultado.
4. Pulsa **LIMPIAR LIENZO** o la barra espaciadora para borrar.
5. El botón **VOZ** silencia o reactiva la lectura.

### 6. Probar el modo Pizarrón

1. Pulsa **PIZARRÓN** en el selector de modo.
2. Acepta el permiso de cámara del navegador.
3. Si hay varias cámaras, elige la correcta en el desplegable **CÁMARA** (por ejemplo, una webcam externa).
4. Coloca la webcam mirando hacia abajo, sobre una pizarra blanca mate.
5. Pulsa **MARCAR ZONA** y arrastra sobre la imagen de la cámara para delimitar el área de la pizarra.
6. Deja la pizarra **vacía** y pulsa **PIZARRA VACÍA**. Esto captura la referencia del fondo.
7. Dibuja con un rotulador negro. La app detectará el trazo y mostrará las predicciones.
8. Si cambias la iluminación, la zona o la cámara, vuelve a capturar la pizarra vacía.

> El modo **Ratón** siempre está disponible como respaldo si la cámara falla.

---

## Solución de problemas

| Problema | Causa probable | Solución |
| --- | --- | --- |
| El modelo no carga o la app queda en modo demo | Se abrió con `file://` o falta la carpeta `model/` | Sirve la carpeta por `http://localhost` y verifica los tres archivos de `model/`. |
| La cámara no enciende | Falta permiso, otra app la usa o el contexto no es seguro | Acepta el permiso, cierra otras apps que usen la cámara y abre por `localhost` o HTTPS. |
| No aparece el desplegable **CÁMARA** | Se estaba en modo ratón o el archivo quedó en caché | Elige **PIZARRÓN** y recarga con `Ctrl + F5`. |
| "Calibración caducada" | La referencia guardada ya no coincide con la zona o la luz | Pulsa **PIZARRA VACÍA** de nuevo. |
| La cámara enfoca la equivocada | El navegador eligió la integrada | Selecciona la webcam externa en el desplegable **CÁMARA**; la elección se recuerda. |
| Se ve el trazo pero no se predice nada | Umbral o referencia de fondo inadecuados | Afina con `tools/prueba-camara.html` y vuelve a capturar la pizarra vacía. |
| El `.bat` dice que falta `caddy.exe` | No se copió el servidor a la carpeta | Descarga Caddy y coloca `caddy.exe` junto a `INICIAR.bat`. |

### Página de afinado de cámara

Abre `http://localhost:8000/tools/prueba-camara.html` para ajustar en vivo:

- Umbral de tinta.
- Vecinos mínimos.
- Tamaño de muestreo.
- Zona de la pizarra.
- Captura de la pizarra vacía.

Es una herramienta de desarrollo: no es necesaria para usar la app final.

---

## Clases reconocidas (24)

```text
casa, sol, arbol, nube, estrella, luna, flor, palmera, manzana,
triangulo, cara, ojo, gato, perro, pez, pajaro, mariposa, coche,
avion, barco, bicicleta, montana, guitarra, reloj
```

El orden y los nombres se leen de `model/class_names.json`, que es la fuente autoritativa.

---

## Notas técnicas

- **Modelo:** TensorFlow.js `graph-model` convertido desde un modelo de Keras/TensorFlow.
- **Entrada:** imagen de `28 × 28 × 1` en escala de grises, normalizada a `[0, 1]`.
- **Preprocesamiento:** la imagen de dibujo (`500 × 500`) o la máscara de la cámara se recortan al bounding box del trazo, se centran y se escalan a `28 × 28`.
- **Cámara:** la máscara binaria de tinta (negro sobre blanco) entra al mismo pipeline que el lienzo del ratón.
- **Estabilidad:** histéresis por firma de máscara para evitar predicciones temblorosas.
- **Persistencia:** el modo de entrada, la cámara elegida y la calibración se guardan en `localStorage`.
- **Offline:** TensorFlow.js, las fuentes y el modelo están en el propio proyecto.

---

## Créditos

Hecho por **Alex Contreras** y **Jasneidy Pernia** para el **XXVI Encuentro con la Física, Química, Matemática y Biología**.

Proyecto con fines educativos y de divulgación científica.
