/* ==========================================================================
   AdivinIA: El Ojo de la Máquina  —  Lógica de la aplicación
   --------------------------------------------------------------------------
   Estructura del archivo:
     1. CONFIGURACIÓN  (todo lo que debes personalizar está aquí arriba)
     2. REFERENCIAS AL DOM
     3. ESTADO DE LA APLICACIÓN
     4. ARRANQUE (init)
     5. LÓGICA DEL LIENZO (dibujar con el ratón / el dedo)
     6. PREPROCESAMIENTO (500x500  ->  28x28 grises normalizado 0..1)
     7. VISUALIZADOR 28x28 (lo que la IA realmente ve)
     8. PREDICCIÓN (tfjs) y actualización de las barras Top 3
     9. CARGA DEL MODELO
    10. UTILIDADES DE INTERFAZ (avisos, estados, atajos de teclado)
   ========================================================================== */

(function () {
    "use strict";

    /* ======================================================================
       1. CONFIGURACIÓN  ★★★ MODIFICA SOLO ESTE BLOQUE ★★★
       ====================================================================== */

    /* ------------------------------------------------------------------
       RUTA DE TU MODELO DE TENSORFLOW.JS
       ------------------------------------------------------------------
       Este es EL único valor que tienes que cambiar cuando termines de
       entrenar tu modelo en Google Colab.

       En Colab/Keras, convierte el modelo con la extensión "tfjs-converter":
           !tensorflowjs_converter \
               --input_format=tfjs_graph_model \
               --output_format=tfjs_graph_model \
               --saved_model_tags=serve \
               --signature_name=serving_default \
               --quantize_bytes=2 \
               /ruta/en/disco/mi_modelo /ruta/en/disco/carpeta/model
       Después copia el contenido de esa carpeta dentro de la carpeta
       "model" de este proyecto, junto a index.html.
       ------------------------------------------------------------------ */
    const MODEL_URL = "./model/model.json";

    /* ------------------------------------------------------------------
       TIPO DE MODELO: DETECTADO AUTOMÁTICAMENTE
       ------------------------------------------------------------------
       No lo fijamos a mano. La app lee el campo "format" de model.json y
       elige la carga y la ejecución correctas:

         format: "layers-model"   ->  tf.loadLayersModel()  +  predict()
         format: "tfjs-graph-model" ->  tf.loadGraphModel()  +  executeAsync()

       Tu modelo actual es un Keras 3 (generatedBy: "keras v3.13.2") exportado
       con tensorflowjs_converter, así que es format: "layers-model" y necesita
       loadLayersModel(). Si algún día lo reexportas como graph model, la app
       se adaptationá sola sin tocar una línea.

       Ojo: no sirven indistintamente. Un loadGraphModel() sobre un
       layers-model lanza un error de formato, y al revés. */
    const AUTO_DETECTAR_FORMATO = true;
    const FORMATO_POR_DEFECTO = "layers"; // solo si AUTO_DETECTAR_FORMATO = false

    /* ------------------------------------------------------------------
       LISTA DE CATEGORÍAS
       ------------------------------------------------------------------
       Hay dos fuentes y solo una manda:

       1. model/class_names.json  (si LEER_CLASES_DESDE_JSON = true)
          Es la fuente de verdad: la genera el entrenamiento y garantiza que el
          orden de la pantalla coincide exactamente con el de la red neuronal.
          Editar el array de abajo NO cambiaría nada mientras esto esté activo.

       2. El array CLASES_POR_DEFECTO de más abajo, que se usa como copia de
          seguridad si el fetch falla (por ejemplo, abriendo el HTML con doble
          clic en lugar de usar Live Server, donde fetch de archivos locales
          está bloqueado por el navegador).

       REGLA DE ORO: el índice i de la lista es el índice i de la capa Dense de
       salida. Si reordenas las clases, la IA dirá "gato" cuando dibujes un
       perro. No toques el orden, solo los textos. */
    const LEER_CLASES_DESDE_JSON = true;

    /* Copia de seguridad: deben ser las MISMAS 24 clases y en el MISMO orden
       que model/class_names.json. */
    const CLASES_POR_DEFECTO = [
        "casa", "sol", "arbol", "nube", "estrella", "luna", "flor", "palmera",
        "manzana", "triangulo", "cara", "ojo", "gato", "perro", "pez", "pajaro",
        "mariposa", "coche", "avion", "barco", "bicicleta", "montana",
        "guitarra", "reloj"
    ];

    /* Lista efectiva: se reemplaza por el JSON si se puede leer. */
    let CLASES = CLASES_POR_DEFECTO.slice();

    /* ------------------------------------------------------------------
       NOMBRES PARA LA PANTALLA
       ------------------------------------------------------------------
       Las claves internas van sin tildes y en minúsculas (ASCII puro) para que
       coincidan exactamente con el dataset. Aquí les ponemos tildes y mayúscula
       solo para que se lean bien delante del público de la feria.
       Cualquier categoría que no esté en este mapa se muestra capitalizada
       automáticamente. */
    const NOMBRES_PANTALLA = {
        arbol: "Árbol",
        palmera: "Palmera",
        triangulo: "Triángulo",
        pajaro: "Pájaro",
        avion: "Avión",
        montana: "Montaña"
    };

    /* Pone la primera letra en mayúscula y deja el resto como está */
    function capitalizar(texto) {
        return texto.charAt(0).toUpperCase() + texto.slice(1);
    }

    /* ------------------------------------------------------------------
       GEOMETRÍA DE ENTRADA
       ------------------------------------------------------------------ */
    const TAMANO_LIENZO = 500;   // Resolución real del lienzo (cuadrado)
    const TAMANO_ENTRADA = 28;   // Resolución que espera la red (MNIST)

    /* ------------------------------------------------------------------
       ESTILO DEL TRAZO
       ------------------------------------------------------------------ */
    const COLOR_TRAZO = "#000000";
    const GROSOR_TRAZO = 26;         // px sobre el lienzo de 500x500
    /* Si entrenaste con trazos finos tipo dígitos (MNIST), sube este valor
       a 40-60. Para Quick, Draw! un valor entre 20 y 30 funciona bien. */

    /* ------------------------------------------------------------------
       PREPROCESAMIENTO  ★★★ REVISA ESTAS DOS BANDERAS ★★★
       ------------------------------------------------------------------
       INVERTIR_IMAGEN = true   ->  La IA espera FONDO NEGRO y TRAZO BLANCO
                                  (formato estándar de MNIST / Quick, Draw!).
                                  Lienzo blanco + tinta negra se invierte.
       INVERTIR_IMAGEN = false  ->  La IA espera FONDO BLANCO y TRAZO NEGRO.
                                  Deja los píxeles tal cual. */
    const INVERTIR_IMAGEN = true;

    /* ------------------------------------------------------------------
       RECORTE Y ESCALA (SIEMPRE ACTIVO)
       ------------------------------------------------------------------
       Google Quick, Draw! normalizó sus 50 millones de dibujos así: recortar al
       rectángulo envolvente del trazo, escalar el LADO MÁS LARGO a 20 px
       manteniendo la proporción, y centrarlo en el lienzo de 28x28 (lo que
       deja al menos 4 px de margen en los bordes).

       Hay que replicarlo EXACTAMENTE en la inferencia o la precisión se desploma,
       porque el modelo solo ha visto dibujos con esa distribución.

       NO cambies 20 por otro valor salvo que reentrenes con el mismo valor. */
    const TAMANO_MAX_TRACO = 20;  // lado mayor del dibujo tras escalar
    const UMBRAL_TINTA = 200;     // separa tinta (0) de fondo (255) por luminancia

    /* ------------------------------------------------------------------
       MODO DEMO
       ------------------------------------------------------------------
       Si el modelo todavía no existe en ./model/, pon MODO_DEMO_SIN_MODELO
       en true para SIMULAR predicciones y poder probar la interfaz en la
       feria sin ningún archivo .bin. En cuanto tengas tu modelo, déjalo en
       false para usar la IA real.

       Si ya tienes model.json y los .bin, déjalo en FALSE: si el archivo se
       carga mal por cualquier razón, es preferible ver un error claro a que
       la feria vea datos falsos. */
    const MODO_DEMO_SIN_MODELO = false;

    /* Ruta del archivo de clases que genera el entrenamiento */
    const CLASES_URL = "./model/class_names.json";

    /* Lectura en voz alta del Top 1 (ideal para niños y personas con
       discapacidad visual). Se puede activar/desactivar con el botón VOZ. */
    const VOZ_POR_DEFECTO = false;

    /* ======================================================================
       2. REFERENCIAS AL DOM
       ====================================================================== */
    const $ = (id) => document.getElementById(id);

    const elLienzo = $("canvas-dibujo");       // Lienzo grande (500x500)
    const elVisualizador = $("canvas-visualizador"); // Mini lienzo 28x28
    const elBtnLimpiar = $("btn-limpiar");
    const elBtnVoz = $("btn-voz");
    const elEstado = $("estado-dibujo");
    const elBadge = $("badge-estado");
    const elToastContenedor = $("toast-contenedor");
    const elLatencia = $("tech-latencia");
    const elContenedorPredicciones = $("predicciones-lista");
    const elBrillo1 = $("brillo-1");

    /* Filas de las 3 barras de predicción, agrupadas por su id "pred-N-*" */
    const FILAS = [1, 2, 3].map((n) => ({
        item: document.querySelector(`.prediccion-item[data-rank="${n}"]`),
        nombre: $(`pred-${n}-nombre`),
        porcentaje: $(`pred-${n}-porcentaje`),
        barra: $(`barra-${n}`)
    }));

    /* ======================================================================
       3. ESTADO DE LA APLICACIÓN
       ====================================================================== */
    const ESTADO = {
        modelo: null,          // tf.LayersModel | tf.GraphModel
        formato: "layers",     // "layers" | "graph", detectado del model.json
        cargado: false,        // ¿el modelo real ya está en memoria?
        demo: false,           // ¿estamos simulando predicciones?
        canales: 1,            // 1 (grises) o 3 (RGB)
        formaEntrada: null,    // p. ej. [-1, 28, 28, 1]
        dibujando: false,      // ¿el botón está presionado?
        ultimoX: 0,
        ultimoY: 0,
        puntoActual: null,     // {x, y} del último punto sin procesar
        trazoEnCurso: false,   // hubo trazo desde el último mouseup
        voz: VOZ_POR_DEFECTO,
ultimoGanador: null,     // nombre del Top 1 anterior (para el destello)
        temporizadorVoz: null
    };

    /* Contextos 2D */
    const ctx = elLienzo.getContext("2d", { willReadFrequently: true });
    const ctxVisualizador = elVisualizador.getContext("2d");

    /* Lienzos auxiliares fuera de pantalla (no se ven en pantalla) */
    const aux28 = document.createElement("canvas");
    aux28.width = aux28.height = TAMANO_ENTRADA;
    const ctxAux28 = aux28.getContext("2d", { willReadFrequently: true });

    const salida28 = document.createElement("canvas");
    salida28.width = salida28.height = TAMANO_ENTRADA;
    const ctxSalida28 = salida28.getContext("2d", { willReadFrequently: true });

    /* Para escalar el mini-lienzo sin suavizado (efecto "pixelado") */
    const pixeles28 = document.createElement("canvas");
    pixeles28.width = pixeles28.height = TAMANO_ENTRADA;
    const ctxPixeles28 = pixeles28.getContext("2d");

    /* Imagen de 28x28 en memoria, reutilizada en cada actualización */
    const imagenPixeles = ctxPixeles28.createImageData(TAMANO_ENTRADA, TAMANO_ENTRADA);

    /* ======================================================================
       4. ARRANQUE
       ====================================================================== */
    function init() {
        /* Primero el lienzo en blanco y las barras a cero (todo síncrono),
           y después las cargas asíncronas, para que los mensajes de estado
           de la carga no pisen este texto inicial. */
        limpiarLienzo();
        registrarEventosLienzo();
        registrarEventosUI();

        /* Las clases y el modelo se cargan en paralelo: son independientes y
           así el usuario no espera dos veces. */
        cargarClases();
        cargarModelo();
    }

    /* ======================================================================
       4.1 LISTA DE CLASES DESDE class_names.json
       ======================================================================
       Así el orden de la interfaz queda garantizado por el propio archivo que
       generó el entrenamiento, sin depender de que alguien copie bien el array
       a mano. Si el fetch falla ( Live Server apagado, JSON mal formado) se
       avisa por consola y se sigue con la copia de seguridad. */
    async function cargarClases() {
        if (!LEER_CLASES_DESDE_JSON) {
            actualizarContadorClases();
            return;
        }

        try {
            const respuesta = await fetch(CLASES_URL, { cache: "no-store" });
            if (!respuesta.ok) {
                throw new Error(`HTTP ${respuesta.status} al pedir ${CLASES_URL}`);
            }

            let leidas = await respuesta.json();

            // Aceptamos las dos formas habituales: ["casa", ...] o {"classes": [...]}
            if (!Array.isArray(leidas)) {
                leidas = leidas.classes || leidas.class_names || leidas.labels;
            }
            if (!Array.isArray(leidas) || leidas.length === 0) {
                throw new Error("El JSON no contiene una lista de clases válida.");
            }

            CLASES = leidas.map((c) => String(c).trim());
            console.log("[AdivinIA] Clases cargadas desde class_names.json:", CLASES);
        } catch (error) {
            console.warn("[AdivinIA] No se pudo leer class_names.json:", error.message,
                        "· se usan las clases por defecto.");
            CLASES = CLASES_POR_DEFECTO.slice();
        }

        actualizarContadorClases();
    }

    /* ======================================================================
       5. LÓGICA DEL LIENZO
       ====================================================================== */

    /* Traduce las coordenadas de pantalla a coordenadas reales del lienzo
       (importante porque el CSS escala el canvas de forma responsiva). */
    function obtenerCoordenadas(evento) {
        const rect = elLienzo.getBoundingClientRect();
        const escalaX = elLienzo.width / rect.width;
        const escalaY = elLienzo.height / rect.height;
        return {
            x: (evento.clientX - rect.left) * escalaX,
            y: (evento.clientY - rect.top) * escalaY
        };
    }

    function registrarEventosLienzo() {
        // Usamos Pointer Events: cubre ratón, lápiz y pantalla táctil.
        elLienzo.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            elLienzo.setPointerCapture(e.pointerId);
            ESTADO.dibujando = true;
            ESTADO.trazoEnCurso = true;
            const { x, y } = obtenerCoordenadas(e);
            ESTADO.ultimoX = x;
            ESTADO.ultimoY = y;
            ESTADO.puntoActual = { x, y };
            pintarPunto(x, y); // para que un "clic" sin arrastre también deje marca
            actualizarEstado("Dibujando...", "procesando");
        });

        elLienzo.addEventListener("pointermove", (e) => {
            if (!ESTADO.dibujando) return;
            e.preventDefault();
            const { x, y } = obtenerCoordenadas(e);
            pintarSegmento(ESTADO.ultimoX, ESTADO.ultimoY, x, y);
            ESTADO.ultimoX = x;
            ESTADO.ultimoY = y;
            ESTADO.puntoActual = { x, y };
            ESTADO.trazoEnCurso = true;
        });

        /* >>> EVENTO CLAVE <<<
           Cada vez que el usuario LEVANTA el ratón (mouseup / pointerup)
           se dispara el preprocesamiento y la predicción. Así el procesador
           no se satura y la simulación sigue siendo fluida. */
        const soltar = (e) => {
            if (!ESTADO.dibujando) return;
            e.preventDefault();
            ESTADO.dibujando = false;
            ESTADO.puntoActual = null;

            if (ESTADO.trazoEnCurso) {
                ejecutarPipeline(); // preprocesar -> predecir -> pintar barras
            }
            ESTADO.trazoEnCurso = false;
        };
        elLienzo.addEventListener("pointerup", soltar);
        elLienzo.addEventListener("pointercancel", soltar);

        /* El scroll con el dedo ya está bloqueado por `touch-action: none`
           en el CSS, así que no hace falta interceptar touchstart/touchmove. */
    }

    /* Configuración del pincel (se llama en cada inicio de trazo) */
    function configurarPincel() {
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = GROSOR_TRAZO;
        ctx.strokeStyle = COLOR_TRAZO;
        ctx.fillStyle = COLOR_TRAZO;
    }

    /* Punto suelto (un clic sin arrastrar) */
    function pintarPunto(x, y) {
        configurarPincel();
        ctx.beginPath();
        ctx.arc(x, y, GROSOR_TRAZO / 2, 0, Math.PI * 2);
        ctx.fill();
    }

    /* Tramo entre dos puntos con extremos redondeados: el trazo sale
       continuo y suave aunque el ratón se mueva a saltos. */
    function pintarSegmento(x0, y0, x1, y1) {
        configurarPincel();
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
    }

    function limpiarLienzo() {
        // Lienzo de dibujo: blanco puro
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, elLienzo.width, elLienzo.height);
        ctx.fillStyle = COLOR_TRAZO; // restaura el color de la tinta

        // Visualizador 28x28: se deja todo a 0 (vacío)
        ctxVisualizador.fillStyle = "#061229";
        ctxVisualizador.fillRect(0, 0, elVisualizador.width, elVisualizador.height);
        dibujarRejillaVisualizador();

        ctxPixeles28.clearRect(0, 0, TAMANO_ENTRADA, TAMANO_ENTRADA);

        ESTADO.trazoEnCurso = false;
        ESTADO.ultimoGanador = null;

        // Reinicia las barras de predicción
        actualizarBarras(["—", "—", "—"], [0, 0, 0]);
        actualizarEstado("Lienzo limpio. ¡Dibuja algo!", "");
    }

    /* ======================================================================
       6. PREPROCESAMIENTO
       ======================================================================
       Convierte lo que hay en el canvas grande (500x500, color) en un tensor
       [1, 28, 28, 1] de floats entre 0 y 1, listo para un modelo entrenado
       en Colab/Keras.
       ====================================================================== */

    /* Devuelve un Float32Array de 784 valores normalizados (0..1) y además
       deja el resultado dibujado en `salida28` para el visualizador.
       Devuelve null si el lienzo está en blanco. */
    function preprocesar() {
        const T = TAMANO_ENTRADA;

        /* --- PASO 1: reescalar el lienzo 500x500 -> 28x28 ---
           El lienzo auxiliar mantiene SIEMPRE el fondo blanco y el trazo
           negro (igual que ve el usuario). La inversión de colores se
           aplica al final, en el paso 3, y solo sobre los valores numéricos.
           Aplicarla aquí y también al final sería una doble inversión. */
        ctxAux28.fillStyle = "#ffffff";
        ctxAux28.fillRect(0, 0, T, T);
        ctxAux28.imageSmoothingEnabled = true;
        ctxAux28.imageSmoothingQuality = "high";
        ctxAux28.drawImage(elLienzo, 0, 0, elLienzo.width, elLienzo.height, 0, 0, T, T);

        /* --- PASO 2: recortar al bounding box y escalar el lado mayor a 20 px --- */
        const caja = calcularCajaContenido();
        if (!caja) {
            // Lienzo en blanco: no hay nada que recortar ni que adivinar.
            return null;
        }

        const ancho = caja.maxX - caja.minX + 1;
        const alto = caja.maxY - caja.minY + 1;

        // Un único factor de escala para ambos lados: mantiene la proporción
        const escala = TAMANO_MAX_TRACO / Math.max(ancho, alto);
        const nuevoAncho = ancho * escala;
        const nuevoAlto = alto * escala;

        // Centrado: deja al menos (28 - 20) / 2 = 4 px de margen en cada eje
        const destinoX = (T - nuevoAncho) / 2;
        const destinoY = (T - nuevoAlto) / 2;

        ctxSalida28.fillStyle = "#ffffff";
        ctxSalida28.fillRect(0, 0, T, T);
        ctxSalida28.imageSmoothingEnabled = true;
        ctxSalida28.imageSmoothingQuality = "high";
        ctxSalida28.drawImage(
            aux28,
            caja.minX, caja.minY, ancho, alto,
            destinoX, destinoY, nuevoAncho, nuevoAlto
        );

        /* --- PASO 3: leer píxeles -> escala de grises -> normalizar 0..1 --- */
        const datos = ctxSalida28.getImageData(0, 0, T, T).data;
        const valores = new Float32Array(T * T);

        for (let i = 0; i < valores.length; i++) {
            const p = i * 4;
            // Luminancia ponderada (fórmula estándar de imagen en color a gris)
            const gris = 0.299 * datos[p] + 0.587 * datos[p + 1] + 0.114 * datos[p + 2];

            if (INVERTIR_IMAGEN) {
                // Formato MNIST / Quick, Draw!: fondo -> 0.0 ; trazo -> 1.0
                valores[i] = (255 - gris) / 255;
            } else {
                // Formato "papel": fondo -> 1.0 ; trazo -> 0.0
                valores[i] = gris / 255;
            }
        }

        return valores;
    }

    /* Busca el rectángulo que contiene toda la tinta (bounding box). */
    /* Busca el rectángulo que contiene toda la tinta (bounding box).
       Se trabaja sobre `aux28`, que siempre tiene fondo BLANCO y trazo
       NEGRO, así que "hay tinta" es simplemente "el píxel es oscuro". */
    function calcularCajaContenido() {
        const T = TAMANO_ENTRADA;
        const datos = ctxAux28.getImageData(0, 0, T, T).data;
        let minX = T, minY = T, maxX = -1, maxY = -1;

        for (let y = 0; y < T; y++) {
            for (let x = 0; x < T; x++) {
                const p = (y * T + x) * 4;
                const gris = 0.299 * datos[p] + 0.587 * datos[p + 1] + 0.114 * datos[p + 2];
                if (gris < UMBRAL_TINTA) { // píxel oscuro = hay tinta
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        return maxX < 0 ? null : { minX, minY, maxX, maxY };
    }

    /* Convierte el array de 784 valores en el tensor que espera la red. */
    function crearTensor(valores) {
        const T = TAMANO_ENTRADA;
        // El tensor lleva dimension de batch: [1, 28, 28, canales]
        const tensor = tf.tensor4d(valores, [1, T, T, 1], "float32");

        // Si el modelo esperaba 3 canales (RGB), replicamos el gris en R, G y B.
        if (ESTADO.canales === 3) {
            const rgbs = new Float32Array(T * T * 3);
            for (let i = 0; i < valores.length; i++) {
                rgbs[i * 3] = valores[i];
                rgbs[i * 3 + 1] = valores[i];
                rgbs[i * 3 + 2] = valores[i];
            }
            tensor.dispose();
            return tf.tensor4d(rgbs, [1, T, T, 3], "float32");
        }
        return tensor;
    }

    /* ======================================================================
       7. VISUALIZADOR 28x28  (divulgación científica)
       ====================================================================== */
    function pintarVisualizador(valores) {
        if (!valores) {
            limpiarVisualizador();
            return;
        }
        const T = TAMANO_ENTRADA;
        const imagen = imagenPixeles;

        for (let i = 0; i < valores.length; i++) {
            const v = valores[i];                 // 0 = fondo, 1 = trazo
            const p = i * 4;
            // Tinte "blueprint": de azul tinta (fondo) a cian (trazo).
            // Para ver gris puro, cambia las constantes por g = v*255.
            imagen.data[p]     = 6 + v * (77 - 6);   // R: 6   -> 77
            imagen.data[p + 1] = 18 + v * (216 - 18); // G: 18  -> 216
            imagen.data[p + 2] = 41 + v * (255 - 41); // B: 41  -> 255
            imagen.data[p + 3] = 255;
        }
        ctxPixeles28.putImageData(imagen, 0, 0);

        // Ampliamos los 28x28 a 280x280 SIN suavizado: se ve cada píxel.
        const lado = elVisualizador.width;
        ctxVisualizador.imageSmoothingEnabled = false;
        ctxVisualizador.clearRect(0, 0, lado, lado);
        ctxVisualizador.drawImage(pixeles28, 0, 0, lado, lado);

        dibujarRejillaVisualizador();
    }

    /* Cuadriculado de 10x10 (= 1 píxel del tensor) para que se entienda
       la escala a la que realmente trabaja la red. */
    function dibujarRejillaVisualizador() {
        const lado = elVisualizador.width;
        const paso = lado / TAMANO_ENTRADA;
        ctxVisualizador.save();
        ctxVisualizador.strokeStyle = "rgba(120, 180, 255, 0.22)";
        ctxVisualizador.lineWidth = 1;
        ctxVisualizador.beginPath();
        for (let i = 1; i < TAMANO_ENTRADA; i++) {
            ctxVisualizador.moveTo(i * paso, 0);
            ctxVisualizador.lineTo(i * paso, lado);
            ctxVisualizador.moveTo(0, i * paso);
            ctxVisualizador.lineTo(lado, i * paso);
        }
        ctxVisualizador.stroke();

        // Ejes X e Y más marcados (como en un plano técnico)
        ctxVisualizador.strokeStyle = "rgba(77, 216, 255, 0.55)";
        ctxVisualizador.lineWidth = 1.5;
        ctxVisualizador.strokeRect(0, 0, lado, lado);
        ctxVisualizador.restore();
    }

    function limpiarVisualizador() {
        ctxPixeles28.clearRect(0, 0, TAMANO_ENTRADA, TAMANO_ENTRADA);
        ctxVisualizador.fillStyle = "#061229";
        ctxVisualizador.fillRect(0, 0, elVisualizador.width, elVisualizador.height);
        dibujarRejillaVisualizador();
    }

    /* ======================================================================
       8. PREDICCIÓN
       ====================================================================== */

    /* Pipeline completo: preprocesar -> mostrar tensor -> predecir -> barras */
    async function ejecutarPipeline() {
        try {
            actualizarEstado("Preprocesando imagen...", "procesando");
            elContenedorPredicciones.classList.add("calculando");

            const valores = preprocesar();

            if (!valores) {
                // Lienzo en blanco: no hay nada que adivinar.
                actualizarEstado("Dibuja algo en el lienzo para que la IA vea algo.", "");
                return;
            }

            pintarVisualizador(valores);

            const t0 = performance.now();
            let probabilidades;
            let latencia;

            if (ESTADO.cargado && ESTADO.modelo) {
                const resultado = await predecirConModelo(valores);
                probabilidades = resultado.probabilidades;
                latencia = resultado.latencia;
            } else {
                const resultado = predecirDemo(valores);
                probabilidades = resultado.probabilidades;
                latencia = resultado.latencia;
            }

            elLatencia.textContent = `${latencia.toFixed(1)} ms`;

            const top3 = obtenerTop3(probabilidades);
            actualizarBarras(top3.map((t) => t.nombre),
                             top3.map((t) => t.probabilidad));
            hablar(top3[0]);

            if (ESTADO.cargado) {
                actualizarEstado(`Analizado en ${latencia.toFixed(1)} ms · ${TAMANO_ENTRADA}×${TAMANO_ENTRADA} = ${TAMANO_ENTRADA * TAMANO_ENTRADA} píxeles`, "listo");
            } else {
                actualizarEstado("MODO DEMO: predicciones simuladas (falta ./model/model.json)", "");
            }
        } catch (error) {
            console.error("[AdivinIA] Error en el pipeline:", error);
            actualizarEstado("Ocurrió un error al analizar el dibujo.", "error");
            mostrarToast(`Error: ${error.message}`, "error");
        } finally {
            elContenedorPredicciones.classList.remove("calculando");
        }
    }

    /* Paso de inferencia real con TensorFlow.js.
       El método de ejecución depende del formato del modelo (detectado en
       cargarModelo): predict() para LayersModel, executeAsync() para GraphModel.
       En ambos casos se ejecuta fuera del hilo principal de forma asíncrona, así
       que el lienzo sigue respondiendo mientras la IA razona. */
    async function predecirConModelo(valores) {
        if (!ESTADO.modelo) {
            throw new Error("El modelo todavía no está cargado.");
        }

        const tensor = crearTensor(valores);
        let tensorSalida = null;

        try {
            const t0 = performance.now();

            /* Cada tipo de modelo se ejecuta de una manera:
               - LayersModel (tu caso): predict() devuelve un Tensor, o un array
                 de tensores si el modelo tiene varias salidas.
               - GraphModel: executeAsync() devuelve un objeto con los tensores
                 indexados por nombre, p. ej. { "StatefulPartitionedCall:0": T }.
               extraerTensorSalida() normaliza ambos casos a un solo tensor. */
            const resultado = ESTADO.formato === "layers"
                ? await ESTADO.modelo.predict(tensor)
                : await ESTADO.modelo.executeAsync(tensor);

            tensorSalida = extraerTensorSalida(resultado);

            if (!tensorSalida) {
                throw new Error("El modelo no devolvió ninguna salida de probabilidades.");
            }

            // dataSync() lee los valores al instante: válido porque el tensor ya
            // está resuelto y es diminuto (24 floats).
            const probabilidades = tensorSalida.dataSync();
            const latencia = performance.now() - t0;

            return { probabilidades, latencia };
        } catch (error) {
            /* Si el modelo esperaba 3 canales (RGB) y le damos 1 (o al revés),
               se reintenta una sola vez con la otra configuración. */
            if (ESTADO.canales === 1) {
                console.warn("[AdivinIA] Reintentando la predicción con 3 canales (RGB):", error.message);
                ESTADO.canales = 3;
                return predecirConModelo(valores);
            }
            throw error;
        } finally {
            /* Liberamos la memoria de la GPU SIEMPRE, aunque hayaSaltado una
               excepción. Es lo que evita que la laptop se llene de tensores
               huérfanos a lo largo de una hora de feria.
               OJO: hay que liberar los TENSORES, no el objeto contenedor:
               tf.dispose() acepta un tensor o un array de tensores. */
            tensor.dispose();
            if (tensorSalida) tensorSalida.dispose();
        }
    }

    /* Saca el tensor de probabilidades de lo que devuelve predict() o
       executeAsync(), sea un tensor suelto, un array o un objeto con los
       nombres de salida.

       Un Graph Model exportado de Colab puede devolver VARIAS salidas (por
       ejemplo, si la firma incluye métricas), así que no nos quedamos a ciegas
       con la primera: buscamos la que tenga exactamente un valor por clase. */
    function extraerTensorSalida(resultado) {
        let tensores;
        if (resultado instanceof tf.Tensor) {
            tensores = [resultado];
        } else if (Array.isArray(resultado)) {
            tensores = resultado.filter((t) => t instanceof tf.Tensor);
        } else if (resultado && typeof resultado === "object") {
            tensores = Object.values(resultado).filter((t) => t instanceof tf.Tensor);
        } else {
            return null;
        }

        if (tensores.length === 0) return null;

        // Preferimos la salida con una probabilidad por clase
        const buena = tensores.find(
            (t) => t.size === CLASES.length || t.dataSync().length === CLASES.length
        );
        return buena || tensores[0];
    }

    /* Simulación de predicciones para probar la interfaz sin modelo.
       ESTO NO ES IA REAL: son números aleatorios con forma de distribución. */
    function predecirDemo(valores) {
        console.info("[AdivinIA] MODO DEMO activo: las probabilidades son simuladas.");

        // Usamos la "cantidad de tinta" para que la simulación cambie al
        // redibujar y no parezca siempre igual.
        let tinta = 0;
        for (let i = 0; i < valores.length; i++) tinta += valores[i];
        const semilla = Math.floor(tinta * 100) + valores.length;

        const pesos = CLASES.map((_, i) => {
            const r = Math.abs(Math.sin(semilla * 0.0007 + i * 2.399));
            return Math.pow(r, 6) + 0.001; // distribución desigual estilo softmax
        });
        const suma = pesos.reduce((a, b) => a + b, 0);

        const t0 = performance.now();
        const probabilidades = pesos.map((p) => p / suma);
        const latencia = performance.now() - t0;

        return { probabilidades, latencia };
    }

    /* Devuelve las 3 probabilidades más altas, ordenadas de mayor a menor */
    function obtenerTop3(probabilidades) {
        const indices = [];

        for (let i = 0; i < probabilidades.length; i++) {
            indices.push({ indice: i, probabilidad: probabilidades[i] });
        }
        indices.sort((a, b) => b.probabilidad - a.probabilidad);

        return indices.slice(0, 3).map((item) => ({
            indice: item.indice,
            // La clave interna va sin tildes ("corazon"); aquí se muestra el
            // nombre con tildes y mayúscula ("Corazón"). Si el modelo tiene más
            // salidas que CLASES, no rompemos la vista.
            nombre: CLASES[item.indice]
                ? (NOMBRES_PANTALLA[CLASES[item.indice]] || capitalizar(CLASES[item.indice]))
                : `Clase ${item.indice}`,
            probabilidad: item.indice < CLASES.length ? item.probabilidad : 0
        }));
    }

    /* Pinta nombres, porcentajes y anchos de las barras */
    function actualizarBarras(nombres, probabilidades) {
        FILAS.forEach((fila, i) => {
            const pct = probabilidades[i] || 0;
            const porcentaje = pct * 100;

            fila.nombre.textContent = nombres[i] ?? "—";
            fila.porcentaje.textContent = porcentaje >= 0.05
                ? `${porcentaje.toFixed(1)}%`
                : "0.0%";
            fila.barra.style.width = `${Math.max(0, Math.min(100, porcentaje))}%`;

            if (i === 0) {
                // Destello cuando la IA cambia de opinión
                if (ESTADO.ultimoGanador && ESTADO.ultimoGanador !== nombres[0]) {
                    reiniciarAnimacion(fila.item, "cambio");
                }
                ESTADO.ultimoGanador = nombres[0];
                // El barrido de luz solo corre en la barra ganadora
                elBrillo1.classList.add("animando");
            }
        });
    }

    /* Reinicia una animación CSS quitando y volviendo a poner la clase */
    function reiniciarAnimacion(elemento, clase) {
        elemento.classList.remove(clase);
        void elemento.offsetWidth; // fuerza el reflow
        elemento.classList.add(clase);
    }

    /* ======================================================================
       9. CARGA DEL MODELO
       ====================================================================== */
    async function cargarModelo() {
        try {
            // --- Comprobación 1: ¿está el archivo tf.min.js en la carpeta? ---
            if (typeof tf === "undefined") {
                throw new Error(
                    'No se encontró TensorFlow.js. Coloca el archivo "tf.min.js" ' +
                    "en la misma carpeta que index.html."
                );
            }

            actualizarEstado("Iniciando TensorFlow.js...", "procesando");
            cambiarBadge("CARGANDO MODELO", "cargando");

            // Elegimos la GPU (WebGL) si existe; si no, CPU. Todo local.
            try {
                if (tf.getBackend() !== "webgl") {
                    await tf.setBackend("webgl");
                }
            } catch (e) {
                console.warn("[AdivinIA] WebGL no disponible, se usará la CPU:", e);
                await tf.setBackend("cpu");
            }
            await tf.ready();
            console.log("[AdivinIA] Backend:", tf.getBackend());

            /* --- Comprobación 2: se puede abrir el modelo en MODEL_URL? ---
               Primero leemos el campo "format" del JSON para saber con qué
               función hay que cargarlo, y luego lo cargamos. */
            actualizarEstado("Cargando pesos del modelo...", "procesando");
            ESTADO.formato = await detectarFormatoModelo();
            console.log("[AdivinIA] Formato del modelo:", ESTADO.formato);

            ESTADO.modelo = ESTADO.formato === "layers"
                ? await tf.loadLayersModel(MODEL_URL)
                : await tf.loadGraphModel(MODEL_URL);

            // Inspeccionamos la forma de entrada para saber cuántos canales usa
            ESTADO.formaEntrada = leerFormaEntrada(ESTADO.modelo);
            ESTADO.canales = (ESTADO.formaEntrada && ESTADO.formaEntrada.length === 4)
                ? (ESTADO.formaEntrada[3] || 1)
                : 1;
            ESTADO.cargado = true;

            /* --- Comprobación 3: inferencia de calentamiento + control de coherencia ---
               La primera predicción suele tardar más (compila los kernels).
               Aprovechamos para medir cuántas salidas da el modelo de verdad y
               compararlo con la lista de clases. Si no coinciden, la IA dirá
               mementeredices con el nombre de otra clase, así que hay que
               avisar MUY visiblemente. */
            let numeroSalidas = null;
            try {
                const prueba = await predecirConModelo(
                    new Float32Array(TAMANO_ENTRADA * TAMANO_ENTRADA)
                );
                numeroSalidas = prueba.probabilidades.length;

                if (numeroSalidas !== CLASES.length) {
                    console.error(
                        `[AdivinIA] DESAJUSTE: el modelo devuelve ${numeroSalidas} ` +
                        `probabilidades pero hay ${CLASES.length} clases. ` +
                        `Arregla class_names.json para que coincidan.`
                    );
                    mostrarToast(
                        `Ojo: ${numeroSalidas} salidas del modelo vs ${CLASES.length} clases`,
                        "error"
                    );
                }
            } catch (error) {
                console.warn("[AdivinIA] La inferencia de prueba falló:", error.message);
            }

            // --- ¡MODELO LISTO! ---
            cambiarBadge("IA LISTA", "listo");
            actualizarEstado("¡IA Lista para jugar! Dibuja y te adivino.", "listo");
            mostrarToast("IA Lista para jugar", "ok");
            console.log("[AdivinIA] Modelo cargado.", {
                entrada: ESTADO.formaEntrada,
                clases: CLASES.length,
                salidas: numeroSalidas,
                backend: tf.getBackend()
            });
        } catch (error) {
            console.error("[AdivinIA] No se pudo cargar el modelo:", error);
            ESTADO.cargado = false;
            ESTADO.modelo = null;

            if (MODO_DEMO_SIN_MODELO) {
                // Modo demo: la interfaz sigue viva para probar en la feria.
                ESTADO.demo = true;
                cambiarBadge("MODO DEMO", "demo");
                actualizarEstado("Modo demo: falta " + MODEL_URL + " · predicciones simuladas", "error");
                mostrarToast("Modelo no encontrado: entro en MODO DEMO", "info");
            } else {
                cambiarBadge("ERROR", "error");
                /* El mensaje corto va en pantalla; la instrucción completa,
                   con los comandos de Colab, va a la consola. */
                const corto = error.corto || error.message;
                actualizarEstado(corto, "error");
                mostrarToast(corto, "error");
                console.error("[AdivinIA] INSTRUCCIONES COMPLETAS:\n" + error.message);
            }
        }
    }

    /* Lee el campo "format" de model.json para decidir cómo cargarlo.
       Un fetch extra de 12 KB es infinitamente más barato que adivinar mal y
       que la app no funcione en la feria.

       De paso detecta el caso incompatible mas frecuente: un modelo de Keras 3
       exportado como "layers-model". TF.js solo entiende el formato de capas de
       Keras 2, asi que la carga falla con un error crptico
       ("An InputLayer should be passed either a batchInputShape..."). Aqui lo
       traduimos a un mensaje que dice exactamente que rehacer. */
    async function detectarFormatoModelo() {
        if (!AUTO_DETECTAR_FORMATO) return FORMATO_POR_DEFECTO;

        let info;
        try {
            const respuesta = await fetch(MODEL_URL, { cache: "no-store" });
            if (!respuesta.ok) {
                throw new Error(`No se pudo leer model.json (HTTP ${respuesta.status}).`);
            }
            info = await respuesta.json();
        } catch (error) {
            // Si no se pudo leer el JSON, loadLayersModel()/loadGraphModel()
            // lanzarán un error mucho más claro que este.
            console.warn("[AdivinIA] No se pudo detectar el formato:", error.message);
            return FORMATO_POR_DEFECTO;
        }

        const formato = info.format;
        const keras = info.modelTopology?.keras_version || "";

        /* El conversor escribe "graph-model" en el campo "format" (esa es la
           cadena real; "tfjs-graph-model" no aparece por ningun sitio en
           tf.min.js). Nos apoyamos ademas en la ausencia de "model_config",
           que es lo que distingue de verdad un grafo de un modelo de capas:
           asi funciona aunque el campo "format" venga vacio o inventado. */
        const esGrafo = formato === "graph-model"
            || formato === "tfjs-graph-model"
            || !info.modelTopology?.model_config;

        if (esGrafo) return "graph";

        if (formato === "layers-model") {
            if (keras.startsWith("3")) {
                const error = new Error(
                    "Este modelo es de Keras 3 y TF.js no sabe leerlo. " +
                    "Vuelve a exportarlo como GRAPH MODEL:\n\n" +
                    "  tf.saved_model.save(modelo, '/content/saved_model_dir')\n\n" +
                    "  !tensorflowjs_converter \\\n" +
                    "      --input_format=tf_saved_model \\\n" +
                    "      --output_format=tfjs_graph_model \\\n" +
                    "      --saved_model_tags=serve \\\n" +
                    "      --signature_name=serving_default \\\n" +
                    "      --quantize_bytes=2 \\\n" +
                    "      /content/saved_model_dir ./model\n\n" +
                    "Después copia model.json y los .bin a la carpeta model/ " +
                    "y valida con: node verificar-modelo.js"
                );
                // Versión corta para el estado bajo el lienzo y el aviso flotante
                error.corto = "Modelo de Keras 3: hay que reexportarlo como Graph Model";
                throw error;
            }
            return "layers"; // Keras 2: este formato sí lo entiende TF.js
        }

        console.warn("[AdivinIA] model.json tiene un 'format' desconocido:", formato);
        return FORMATO_POR_DEFECTO;
    }

    /* Lee la forma del tensor de entrada sin lanzar errores si el modelo
       es un GraphModel sin metadatos. */
    function leerFormaEntrada(modelo) {
        try {
            /* LayersModel: inputs[0] es un tensor y trae la forma directa. */
            const entrada = modelo.inputs && modelo.inputs[0];
            if (entrada && entrada.shape) return entrada.shape;

            /* GraphModel: inputs[0] es solo el NOMBRE del nodo (una cadena), sin
               forma. La unica fuente fiable es la firma que dejó el conversor:
               signature.inputs -> { <clave>: { name, tensorShape } }. */
            const firma = modelo.signature && modelo.signature.inputs;
            if (firma) {
                const primera = Object.values(firma)[0];
                const dims = primera && primera.tensorShape && primera.tensorShape.dim;
                if (Array.isArray(dims) && dims.length) {
                    /* OJO: segun como venga la firma, "size" puede ser numero
                       ("28") o cadena ("-1"). Lo forzamos a Number porque
                       abajo se compara con === 1 / === 3 en modo estricto. */
                    return dims.map((d) => (d.size === undefined || d.size === null
                        ? -1
                        : Number(d.size)));
                }
            }
        } catch (e) {
            console.warn("[AdivinIA] No se pudo leer la forma de entrada:", e);
        }
        return null;
    }

    /* ======================================================================
       10. UTILIDADES DE INTERFAZ
       ====================================================================== */

    /* Evento de los botones y atajos de teclado */
    function registrarEventosUI() {
        elBtnLimpiar.addEventListener("click", limpiarLienzo);

        elBtnVoz.addEventListener("click", () => {
            ESTADO.voz = !ESTADO.voz;
            elBtnVoz.classList.toggle("activo", ESTADO.voz);
            mostrarToast(ESTADO.voz ? "Voz activada" : "Voz desactivada", "info");
            if (ESTADO.voz && "speechSynthesis" in window) {
                falar("Voz activada");
            }
        });

        // Atajos: ESPACIO o la tecla C borran el lienzo
        document.addEventListener("keydown", (e) => {
            const esBoton = document.activeElement && document.activeElement.tagName === "BUTTON";
            if ((e.code === "Space" || e.key.toLowerCase() === "c") && !esBoton) {
                e.preventDefault();
                limpiarLienzo();
            }
        });
    }

    /* Mensaje de estado bajo el lienzo */
    function actualizarEstado(texto, tipo) {
        elEstado.querySelector("span:last-child").textContent = texto;
        elEstado.className = "estado-dibujo" + (tipo ? " " + tipo : "");
    }

    /* Etiqueta de estado junto a "Predicciones en Tiempo Real" */
    function cambiarBadge(texto, tipo) {
        elBadge.textContent = texto;
        elBadge.className = "badge" + (tipo ? " " + tipo : "");
    }

    /* Aviso flotante (ej. "IA Lista para jugar") */
    function mostrarToast(mensaje, tipo = "info") {
        const toast = document.createElement("div");
        toast.className = "toast " + tipo;
        toast.textContent = mensaje;
        elToastContenedor.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("saliendo");
            setTimeout(() => toast.remove(), 400);
        }, 3200);
    }

    /* Lee en voz alta la predicción ganadora */
    function falar(texto) {
        if (!ESTADO.voz || !texto || !("speechSynthesis" in window)) return;
        try {
            window.speechSynthesis.cancel(); // cancela la frase anterior
            const frase = new SpeechSynthesisUtterance("Creo que es: " + texto);
            frase.lang = "es-ES";
            frase.rate = 0.95;
            window.speechSynthesis.speak(frase);
        } catch (e) {
            console.warn("[AdivinIA] Voz no disponible:", e);
        }
    }

    /* Muestra cuántas categorías tiene el array CLASES */
    function actualizarContadorClases() {
        const el = document.getElementById("tech-clases");
        if (el) el.textContent = `${CLASES.length} categorías`;
    }

    /* ======================================================================
       ARRANQUE DE LA APLICACIÓN
       ====================================================================== */
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();