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
    const VOZ_POR_DEFECTO = true;

    /* ------------------------------------------------------------------
       MODO DE ENTRADA
       ------------------------------------------------------------------
       "pizarron" -> la webcam mira una pizarra física y la app lee el dibujo.
       "raton"    -> se dibuja sobre la pantalla (modo de emergencia).

       Arranca en "pizarron" y, si la cámara falla, cae solo a "raton" con un
       aviso: nunca dejes la feria con la pantalla en negro. */
    const MODO_ENTRADA_POR_DEFECTO = "pizarron";

    /* ------------------------------------------------------------------
       CÁMARA / VISIÓN  ★★★ AQUÍ SE AFINA EL MONTAJE ★★★
       ------------------------------------------------------------------
       Estos valores son los que probablemente tengas que retocar con tu
       pizarra y tu luz reales. La página prueba-camara.html sirve justo para
       encontrar los buenos sin tocar la app principal.

       - resolucion    : resolución de captura de la webcam.
       - fpsAnalisis   : predicciones por segundo. NO lo pongas a 30: el
                         resultado parpadearía y la voz no pararía de hablar.
       - ladoAnalisis  : el recuadro de la pizarra se reduce a esto (píxeles)
                         antes de analizarlo. Más = más fino pero más lento.
       - umbralTinta   : cuánto más oscuro que la pizarra vacía tiene que ser
                         un píxel para contar como rotulador. Súbelo si se
                         cuela ruido; bájalo si no detecta trazos flojos.
       - umbralMinTinta: fracción mínima del recuadro con tinta para considerar
                         que HAY dibujo (evita disparar con una mota).
       - vecinosMinimos: limpieza de ruido; un píxel de tinta necesita al menos
                         estos vecinos de tinta para sobrevivir.
       - framesEstables : cuántos análisis seguidos debe ganar la misma clase
                         antes de darla por buena (estabilidad).
       - framesReferencia: frames que se promedian al capturar la pizarra vacía. */
    const CAMARA = {
        resolucion:       { width: 640, height: 480 },
        fpsAnalisis:      5,
        ladoAnalisis:     160,
        umbralTinta:      45,
        umbralMinTinta:   0.004,
        vecinosMinimos:   2,
        dilatacion:       1,     // engorda el trazo para acercarlo a Quick, Draw!
        framesEstables:   3,
        framesReferencia: 12
    };

    /* Claves de localStorage: la calibración y el modo sobreviven a un F5.
       En una feria, perder la calibración por recargar la página sería un drama. */
    const CLAVE_LS_CALIBRACION = "adivinIA.calibracion.v1";
    const CLAVE_LS_MODO = "adivinIA.modoEntrada.v1";
    const CLAVE_LS_CAMARA = "adivinIA.camaraId.v1";

    /* ======================================================================
       2. REFERENCIAS AL DOM
       ====================================================================== */
    const $ = (id) => document.getElementById(id);

    const elLienzo = $("canvas-dibujo");       // Lienzo grande (500x500)
    const elVisualizador = $("canvas-visualizador"); // Mini lienzo 28x28
    const elBtnLimpiar = $("btn-limpiar");
    const elBtnVoz = $("btn-voz");

    /* Modo de entrada (ratón / pizarrón) y cámara */
    const elBtnModoPizarron = $("btn-modo-pizarron");
    const elBtnModoRaton = $("btn-modo-raton");
    const elBtnCalibrarZona = $("btn-calibrar-zona");
    const elBtnCalibrarFondo = $("btn-calibrar-fondo");
    const elBtnCambiarCamara = $("btn-cambiar-camara");
    const elContenedorCamara = $("camara-contenedor");
    const elContenedorLienzo = $("lienzo-contenedor");
    const elVideoCamara = $("video-camara");
    const elCanvasCamara = $("canvas-camara");
    const elCamaraAviso = $("camara-aviso");
    const elCamaraAvisoTexto = $("camara-aviso-texto");
    const elCanvasProcesado = $("canvas-procesado");
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
        temporizadorVoz: null,
        calculando: false,       // ¿hay una predicción en marcha?
        rehacer: false,          // ¿llegó otro dibujo mientras se predecía?
        vozUltimaFrase: null,    // frase dicha hace nada (antirrebote)
        vozUltimoMomento: 0,
        modoEntrada: MODO_ENTRADA_POR_DEFECTO,
        stream: null,
        camaras: [],
        indiceCamara: 0,
        camaraLista: false,
        calibracion: null,
        referencia: null,
        marcandoZona: false,
        rectTemporal: null,
        capturandoReferencia: false,
        valoresPendientes: undefined,
        firmaMascara: null,
        firmaEstable: 0,
        firmaPredicha: null
    };

    /* Contextos 2D */
    const ctx = elLienzo.getContext("2d", { willReadFrequently: true });
    const ctxVisualizador = elVisualizador.getContext("2d");
    const ctxCamara = elCanvasCamara.getContext("2d");
    const ctxProcesado = elCanvasProcesado.getContext("2d");

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
        registrarEventosCamara();
        registrarEventosUI();

        // Recupera zona vigilada y pizarra de referencia de sesiones anteriores
        cargarCalibracion();

        /* La voz arranca encendida, así que el botón debe verse ya marcado como
           activo; si no, el rótulo miente. */
        elBtnVoz.classList.toggle("activo", ESTADO.voz);
        elBtnVoz.setAttribute("aria-pressed", String(ESTADO.voz));

        /* Las clases y el modelo se cargan en paralelo: son independientes y
           así el usuario no espera dos veces. */
        cargarClases();
        cargarModelo();

        /* Arranca en el modo guardado: pizarrón con fallback. */
        try {
            const modo = leerModoGuardado();
            cambiarModoEntrada(modo).catch((err) => {
                console.warn("[AdivinIA] No se pudo arrancar en modo pizarrón; paso a ratón:", err.message);
                cambiarModoEntrada("raton").catch(() => {});
            });
        } catch (err) {
            console.warn("[AdivinIA] No se pudo leer el modo guardado:", err.message);
        }
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
            if (ESTADO.modoEntrada !== "raton") return;
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

        // También limpiamos el mini preview procesado
        if (ctxProcesado) {
            ctxProcesado.fillStyle = "#ffffff";
            ctxProcesado.fillRect(0, 0, elCanvasProcesado.width, elCanvasProcesado.height);
        }

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
       Devuelve null si la imagen está en blanco.

       Antes recibía el lienzo por variable global; ahora acepta CUALQUIER
       canvas con fondo blanco y trazo negro. Así el mismo preprocesado sirve
       para el modo ratón (elLienzo) y para el modo pizarrón (la máscara que
       sale de la cámara), sin duplicar ni una línea. */
    function normalizarLienzo(origen) {
        const T = TAMANO_ENTRADA;

        /* --- PASO 1: reescalar la imagen de origen -> 28x28 ---
           El lienzo auxiliar mantiene SIEMPRE el fondo blanco y el trazo
           negro (igual que ve el usuario). La inversión de colores se
           aplica al final, en el paso 3, y solo sobre los valores numéricos.
           Aplicarla aquí y también al final sería una doble inversión. */
        ctxAux28.fillStyle = "#ffffff";
        ctxAux28.fillRect(0, 0, T, T);
        ctxAux28.imageSmoothingEnabled = true;
        ctxAux28.imageSmoothingQuality = "high";
        ctxAux28.drawImage(origen, 0, 0, origen.width, origen.height, 0, 0, T, T);

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

    /* Modo ratón: el origen es el lienzo en el que dibuja el usuario. */
    function preprocesar() {
        return normalizarLienzo(elLienzo);
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
    function crearTensor(valores, canales) {
        const T = TAMANO_ENTRADA;
        const n = canales || 1;

        // El tensor lleva dimension de batch: [1, 28, 28, n]
        if (n === 3) {
            // Si el modelo esperaba 3 canales (RGB), replicamos el gris en R, G y B.
            const rgbs = new Float32Array(T * T * 3);
            for (let i = 0; i < valores.length; i++) {
                rgbs[i * 3] = valores[i];
                rgbs[i * 3 + 1] = valores[i];
                rgbs[i * 3 + 2] = valores[i];
            }
            return tf.tensor4d(rgbs, [1, T, T, 3], "float32");
        }
        return tf.tensor4d(valores, [1, T, T, 1], "float32");
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

    /* Control de concurrencia. Si el usuario levanta el puntero dos veces muy
       seguidas, se solapan dos predicciones sobre el mismo modelo y los tensores
       de una se liberan mientras la otra los sigue usando. En vez de solaparlas
       las encadenamos: si llega otra mientras predicament, se marca una
       "pasada pendiente" y se vuelve a lanzar al terminar. Así siempre se
       muestra la predicción del ÚLTIMO dibujo, que es lo que espera el usuario. */
    async function ejecutarPipeline(valoresEntrada) {
        /* Guardamos los últimos valores recibidos: si llega uno nuevo mientras
           se está calculando, el bucle de abajo retoma ESTE, no el viejo. */
        ESTADO.valoresPendientes = valoresEntrada;

        if (ESTADO.calculando) {
            ESTADO.rehacer = true;
            return;
        }

        ESTADO.calculando = true;
        try {
            do {
                ESTADO.rehacer = false;
                await ejecutarPipelineUnaVez(ESTADO.valoresPendientes);
                ESTADO.valoresPendientes = undefined;
            } while (ESTADO.rehacer);
        } finally {
            ESTADO.calculando = false;
            ESTADO.rehacer = false;
        }
    }

    /* Pipeline completo: preprocesar -> mostrar tensor -> predecir -> barras.
       `valoresEntrada` es opcional: si viene de la cámara ya está calculado;
       si no, se preprocesa el lienzo de dibujo (modo ratón). */
    async function ejecutarPipelineUnaVez(valoresEntrada) {
        try {
            actualizarEstado("Preprocesando imagen...", "procesando");
            elContenedorPredicciones.classList.add("calculando");

            const valores = valoresEntrada || preprocesar();

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
                /* Solo se consolida el nº de canales si la predicción
                   funcionó, para no dejar el estado corrupto. */
                ESTADO.canales = resultado.canales;
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
            /* Antes se mostraba SIEMPRE "Ocurrió un error al analizar el
               dibujo", que no dice nada: si algo falla, hay que poder ver por
               qué sin abrir la consola. Recortamos el texto para que quepa en
               la línea de estado y lo guardamos para el aviso flotante. */
            const motivo = (error && error.message) ? error.message : String(error);
            actualizarEstado(`Error al analizar: ${motivo}`, "error");
            mostrarToast(`Error: ${recortar(motivo, 90)}`, "error");
        } finally {
            elContenedorPredicciones.classList.remove("calculando");
        }
    }

    /* Recorta un texto para que quepa en un aviso sin partirlo por la mitad */
    function recortar(texto, max) {
        return texto.length > max ? texto.slice(0, max - 1).trimEnd() + "…" : texto;
    }

    /* Paso de inferencia real con TensorFlow.js.
       El método de ejecución depende del formato del modelo (detectado en
       cargarModelo): predict() para LayersModel, executeAsync() para GraphModel.
       En ambos casos se ejecuta fuera del hilo principal de forma asíncrona, así
       que el lienzo sigue respondiendo mientras la IA razona. */
    async function predecirConModelo(valores, canales) {
        if (!ESTADO.modelo) {
            throw new Error("El modelo todavía no está cargado.");
        }

        /* Los canales se pasan como ARGUMENTO, no se tocan directamente. Antes
           el reintentoRGB mutaba ESTADO.canales y lo dejaba en 3 para siempre:
           un solo fallo deja la app estropeada de forma permanente. */
        const n = canales || ESTADO.canales || 1;
        const tensor = crearTensor(valores, n);
        let tensorSalida = null;
        let resultado = null;

        try {
            const t0 = performance.now();

            /* Cada tipo de modelo se ejecuta de una manera:
               - LayersModel: predict() devuelve un Tensor, o un array de
                 tensores si el modelo tiene varias salidas.
               - GraphModel: executeAsync() devuelve un objeto con los tensores
                 indexados por nombre.
               extraerTensorSalida() normaliza ambos casos a un solo tensor. */
            resultado = ESTADO.formato === "layers"
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

            return { probabilidades, latencia, canales: n };
        } catch (error) {
            /* Solo tiene sentido reintentar por RGB si el error habla de
               canales o de formas. Antes se reintentaba con CUALQUIER fallo
               (una voz sin dispositif, un tensor ya liberado...) y eso solo
              mogaba más. */
            const esProblemaDeForma = /channel|canal|shape|rank|depth|dimensi/i
                .test(error.message || "");

            if (esProblemaDeForma && n === 1) {
                console.warn("[AdivinIA] El modelo no encaja con 1 canal; se reintenta con 3 (RGB).",
                            error.message);
                return predecirConModelo(valores, 3);
            }
            throw error;
        } finally {
            /* Liberamos la memoria de la GPU SIEMPRE, aunque haya saltado una
               excepción. Es lo que evita que la laptop se llene de tensores
               huérfanos a lo largo de una hora de feria.

               OJO: hay que liberar TODOS los tensores de salida, no solo el
               elegido. Un GraphModel puede devolver varias salidas y dejar las
               demás sin liberar hace que la memoria crezca predicción a
               predicción, hasta que el navegador empieza a fallar sin motivo
               aparente. */
            if (resultado) {
                listarTensors(resultado).forEach((t) => {
                    if (!t.isDisposed) t.dispose();
                });
            } else if (tensorSalida && !tensorSalida.isDisposed) {
                tensorSalida.dispose();
            }
            if (!tensor.isDisposed) tensor.dispose();
        }
    }

    /* Normaliza lo que devuelven predict()/executeAsync() a una lista de tensores:
       un tensor suelto, un array, o un objeto con los nombres de salida. */
    function listarTensors(resultado) {
        if (!resultado) return [];
        if (resultado instanceof tf.Tensor) return [resultado];
        if (Array.isArray(resultado)) {
            return resultado.filter((t) => t instanceof tf.Tensor);
        }
        return Object.values(resultado).filter((t) => t instanceof tf.Tensor);
    }

    /* Saca el tensor de probabilidades de lo que devuelve predict() o
       executeAsync(), sea un tensor suelto, un array o un objeto con los
       nombres de salida.

       Un Graph Model exportado de Colab puede devolver VARIAS salidas (por
       ejemplo, si la firma incluye métricas), así que no nos quedamos a ciegas
       con la primera: buscamos la que tenga exactamente un valor por clase. */
    function extraerTensorSalida(resultado) {
        const tensores = listarTensors(resultado);
        if (tensores.length === 0) return null;

        /* Preferimos la salida con una probabilidad por clase. Solo usamos
           dataSync() si el .size no basta, porque dataSync() es una lectura
           bloqueante y no queremos Repeatedla 20 veces por predicción. */
        const buena = tensores.find((t) => t.size === CLASES.length)
            || tensores.find((t) => {
                try {
                    return t.dataSync().length === CLASES.length;
                } catch (e) {
                    return false;
                }
            });

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

        elBtnLimpiar.addEventListener("click", limpiarLienzo);

        /* Botones de modo de entrada: ratón / pizarrón */
        elBtnModoRaton.addEventListener("click", () => {
            cambiarModoEntrada("raton");
        });
        elBtnModoPizarron.addEventListener("click", () => {
            cambiarModoEntrada("pizarron");
        });

        /* Controles de calibración (modo pizarrón) */
        if (elBtnCalibrarZona) {
            elBtnCalibrarZona.addEventListener("click", marcarZona);
        }
        if (elBtnCalibrarFondo) {
            elBtnCalibrarFondo.addEventListener("click", () => {
                capturarReferencia().catch((err) => {
                    console.error("[AdivinIA] Error capturando referencia:", err);
                    mostrarToast("No se pudo capturar la referencia", "error");
                });
            });
        }
        if (elBtnCambiarCamara) {
            elBtnCambiarCamara.addEventListener("click", () => {
                cambiarCamaraDisponible().catch((err) => {
                    mostrarToast("No se pudo cambiar de cámara", "error");
                    console.warn("[AdivinIA] cambiarCamaraDisponible:", err.message);
                });
            });
        }

        /* La voz ahora viene ACTIVADA por defecto: el botón ya no enciende nada,
           solo silencia. Antes había que pulsarlo para que hablara. */
        elBtnVoz.addEventListener("click", () => {
            ESTADO.voz = !ESTADO.voz;
            elBtnVoz.classList.toggle("activo", ESTADO.voz);
            elBtnVoz.setAttribute("aria-pressed", String(ESTADO.voz));
            if (!ESTADO.voz && "speechSynthesis" in window) {
                window.speechSynthesis.cancel();
                mostrarToast("Voz silenciada", "info");
            } else {
                mostrarToast("Voz activada", "info");
                // Voz de prueba para confirmar que suena sin tener que dibujar
                hablar({ nombre: ESTADO.ultimoGanador || "avion" });
            }
        });

        /* Las voces del sistema llegan de forma asíncrona en algunos
           navegadores. Nos suscribimos para usar una voz española de verdad en
           cuanto estén disponibles. */
        if ("speechSynthesis" in window) {
            window.speechSynthesis.addEventListener("voiceschanged", cargarVozEspanol);
            cargarVozEspanol();
        }

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

    /* Lee en voz alta la predicción ganadora.
       Recibe el objeto del Top 1 ({ nombre, probabilidad }) y usa el nombre
       BONITO de NOMBRES_PANTALLA, que ya lleva tildes: leer "avion" en voz alta
       suena fatal, "Avión" se entiende perfectly. */
    let vozEspanol = null;

    /* Las voces del sistema NO están disponibles en el primer momento: hay que
       esperar al evento "voiceschanged". Si no, el navegador usa la voz por
       defecto del sistema, que puede ser inglesa. */
    function cargarVozEspanol() {
        if (!("speechSynthesis" in window)) return null;
        const voces = window.speechSynthesis.getVoices();
        if (!voces.length) return null;
        vozEspanol = voces.find((v) => /^es[-_]ES/i.test(v.lang))
            || voces.find((v) => /^es/i.test(v.lang))
            || voces[0];
        return vozEspanol;
    }

    function hablar(top) {
        if (!ESTADO.voz || !top || !("speechSynthesis" in window)) return;

        const texto = NOMBRES_PANTALLA[top.nombre] || capitalizar(top.nombre);
        if (!texto) return;

        /* Antirrebote: si se dice exactamente lo mismo en menos de 1,2 s es que
           el usuario sigue rayando encima. Hablar ahí produce un tartamudeo
           que no aporta nada. */
        const ahora = Date.now();
        if (ESTADO.vozUltimaFrase === texto && (ahora - ESTADO.vozUltimoMomento) < 1200) {
            return;
        }
        ESTADO.vozUltimaFrase = texto;
        ESTADO.vozUltimoMomento = ahora;

        try {
            // cancel() corta la frase anterior: si no, las predicciones se
            // apilan y oyes un collage de frases desfasadas.
            window.speechSynthesis.cancel();
            if (!vozEspanol) cargarVozEspanol();

            const frase = new SpeechSynthesisUtterance("Creo que es: " + texto);
            frase.lang = "es-ES";
            if (vozEspanol) frase.voice = vozEspanol;
            frase.rate = 0.95;
            frase.pitch = 1.05;
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
       11. CÁMARA / PIZARRÓN
       ====================================================================== */

    function contextoEsSeguro() {
        return !!window.isSecureContext;
    }

    function mostrarAvisoCamara(texto, esError) {
        if (!elCamaraAviso) return;
        if (texto) {
            elCamaraAviso.hidden = false;
            if (elCamaraAvisoTexto) {
                elCamaraAvisoTexto.textContent = texto;
            }
            elCamaraAviso.classList.toggle("error", !!esError);
        } else {
            elCamaraAviso.hidden = true;
        }
    }

    function ocultarAvisoCamara() {
        mostrarAvisoCamara(null, false);
    }

    async function enumerarCamaras() {
        try {
            const dispositivos = await navigator.mediaDevices.enumerateDevices();
            ESTADO.camaras = dispositivos.filter((d) => d.kind === "videoinput");
        } catch (e) {
            console.warn("[AdivinIA] No se pudieron enumerar las cámaras:", e.message);
            ESTADO.camaras = [];
        }
    }

    async function pedirStreamCamara() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("El navegador no soporta getUserMedia para la cámara.");
        }
        if (!contextoEsSeguro()) {
            throw new Error("Contexto no seguro: abre http://localhost (Live Server) o HTTPS.");
        }

        const restricciones = {
            audio: false,
            video: {
                width:  { ideal: CAMARA.resolucion.width },
                height: { ideal: CAMARA.resolucion.height }
            }
        };

        const id = ESTADO.camaras[ESTADO.indiceCamara]
            && ESTADO.camaras[ESTADO.indiceCamara].deviceId;
        if (id) restricciones.video.deviceId = { exact: id };

        return navigator.mediaDevices.getUserMedia(restricciones);
    }

    function manejarErrorCamara(err) {
        let msg = err && err.message ? err.message : String(err);
        if (/denied|permission|permiso/i.test(msg)) {
            msg = "Permiso de cámara denegado. Actívalo y recarga.";
        } else if (/notfound|not found/i.test(msg)) {
            msg = "No se encontró ninguna cámara.";
        } else if (/secure|contexto/i.test(msg)) {
            msg = "La cámara requiere contexto seguro (http://localhost o HTTPS).";
        }
        mostrarAvisoCamara(msg, true);
        return msg;
    }

    async function iniciarCamara() {
        mostrarAvisoCamara("Encendiendo cámara…", false);

        // La resolución del canvas sigue a la de captura, para dibujar 1:1
        elCanvasCamara.width = CAMARA.resolucion.width;
        elCanvasCamara.height = CAMARA.resolucion.height;

        const stream = await pedirStreamCamara();
        ESTADO.stream = stream;
        elVideoCamara.srcObject = stream;

        /* play() arranca la reproducción. No esperamos al evento
           'loadedmetadata': con el atributo autoplay ese evento ya puede haber
           saltado antes de escuchar, y "esperar un evento ya pasado" deja la
           promesa colgada para siempre (era el "Encendiendo cámara…" eterno).
           En su lugar sondeamos videoWidth, que es lo que de verdad necesitamos. */
        try {
            await elVideoCamara.play();
        } catch (e) {
            console.warn("[AdivinIA] play() rechazó, seguimos igualmente:", e.message);
        }
        await esperarVideoConDatos(elVideoCamara);

        await enumerarCamaras();
        ESTADO.camaraLista = true;
        ocultarAvisoCamara();
        arrancarBucleCamara();
        arrancarAnalisisCamara();
    }

    /* Espera (con tope de tiempo) a que el vídeo tenga frames disponibles.
       Sondear evita la carrera del evento loadedmetadata. */
    async function esperarVideoConDatos(video, timeoutMs = 3000) {
        const inicio = Date.now();
        while (!video.videoWidth && video.readyState < 2) {
            if (Date.now() - inicio > timeoutMs) break;
            await esperar(50);
        }
    }

    function detenerCamara() {
        pararBucleCamara();
        pararAnalisisCamara();
        if (ESTADO.stream) {
            ESTADO.stream.getTracks().forEach((t) => t.stop());
            ESTADO.stream = null;
        }
        if (elVideoCamara) elVideoCamara.srcObject = null;
        ESTADO.camaraLista = false;
    }

    async function cambiarCamaraDisponible() {
        await enumerarCamaras();
        if (ESTADO.camaras.length <= 1) {
            mostrarToast("Solo hay una cámara disponible", "info");
            return;
        }
        ESTADO.indiceCamara = (ESTADO.indiceCamara + 1) % ESTADO.camaras.length;
        detenerCamara();
        await iniciarCamara();
        mostrarToast("Cámara cambiada", "ok");
    }

    /* ----- Bucle de dibujo del frame (Fase 2) -----
       De momento solo pinta la imagen de la cámara en el canvas. En la Fase 4
       aquí mismo se hará la detección de tinta. Lo dejamos ya asentado en un
       bucle propio para no rehacerlo después. */
    let idBucleCamara = null;

    function arrancarBucleCamara() {
        if (idBucleCamara !== null) return;
        const paso = () => {
            idBucleCamara = requestAnimationFrame(paso);
            dibujarFrameCamara();
        };
        paso();
    }

    function pararBucleCamara() {
        if (idBucleCamara !== null) {
            cancelAnimationFrame(idBucleCamara);
            idBucleCamara = null;
        }
    }

    function dibujarFrameCamara() {
        if (!ESTADO.camaraLista || !elVideoCamara.videoWidth) return;
        ctxCamara.drawImage(elVideoCamara, 0, 0, elCanvasCamara.width, elCanvasCamara.height);
        dibujarOverlayCalibracion();
    }

    /* ----- Calibración: zona vigilada (ROI) y pizarra vacía (referencia) -----
       Guardamos las coordenadas en el espacio REAL del canvas (640x480), no en
       píxeles de pantalla, para que sigan valiendo aunque la ventana cambie. */

    function obtenerCoordenadasCamara(evento) {
        const rect = elCanvasCamara.getBoundingClientRect();
        const escalaX = elCanvasCamara.width / rect.width;
        const escalaY = elCanvasCamara.height / rect.height;
        return {
            x: (evento.clientX - rect.left) * escalaX,
            y: (evento.clientY - rect.top) * escalaY
        };
    }

    function rectanguloNormalizado(a, b) {
        return {
            x: Math.round(Math.min(a.x, b.x)),
            y: Math.round(Math.min(a.y, b.y)),
            w: Math.round(Math.abs(a.x - b.x)),
            h: Math.abs(a.y - b.y) >= 0 ? Math.round(Math.abs(a.y - b.y)) : 0
        };
    }

    /* Rectángulo que se analiza: el ROI marcado o, si no hay, todo el frame */
    function regionDeInteres() {
        if (ESTADO.calibracion && ESTADO.calibracion.roi) {
            return ESTADO.calibracion.roi;
        }
        return { x: 0, y: 0, w: elCanvasCamara.width, h: elCanvasCamara.height };
    }

    /* Dibuja la etiqueta de estado y, si se está marcando, el rectángulo */
    function dibujarOverlayCalibracion() {
        if (!ESTADO.marcandoZona) {
            if (ESTADO.referencia) {
                dibujarEtiquetaCamara("FONDO OK", "#37ff9b");
            } else {
                dibujarEtiquetaCamara("SIN FONDO", "#ffb454");
            }
        }

        if (!ESTADO.rectTemporal) return;

        // Oscurecemos todo menos la zona marcada, para ver qué se vigilará
        ctxCamara.save();
        ctxCamara.fillStyle = "rgba(6, 18, 41, 0.55)";
        const c = elCanvasCamara;
        const r = ESTADO.rectTemporal;
        ctxCamara.fillRect(0, 0, c.width, r.y);
        ctxCamara.fillRect(0, r.y + r.h, c.width, c.height - (r.y + r.h));
        ctxCamara.fillRect(0, r.y, r.x, r.h);
        ctxCamara.fillRect(r.x + r.w, r.y, c.width - (r.x + r.w), r.h);

        ctxCamara.strokeStyle = "#4dd8ff";
        ctxCamara.lineWidth = 2;
        ctxCamara.strokeRect(r.x, r.y, r.w, r.h);
        ctxCamara.restore();
    }

    function dibujarEtiquetaCamara(texto, color) {
        ctxCamara.save();
        ctxCamara.font = "bold 13px monospace";
        const ancho = ctxCamara.measureText(texto).width + 16;
        ctxCamara.fillStyle = "rgba(6, 18, 41, 0.82)";
        ctxCamara.fillRect(8, 8, ancho, 24);
        ctxCamara.strokeStyle = color;
        ctxCamara.lineWidth = 1;
        ctxCamara.strokeRect(8, 8, ancho, 24);
        ctxCamara.fillStyle = color;
        ctxCamara.fillText(texto, 16, 25);
        ctxCamara.restore();
    }

    function marcarZona() {
        if (ESTADO.modoEntrada !== "pizarron" || !ESTADO.camaraLista) return;
        ESTADO.marcandoZona = true;
        ESTADO.rectTemporal = null;
        mostrarAvisoCamara("Arrastra sobre la cámara para marcar la zona de la pizarra.", false);
    }

    function registrarEventosCamara() {
        // Mientras se marca la zona, el arrastre vive aquí (no en el lienzo de dibujo)
        elCanvasCamara.addEventListener("pointerdown", (e) => {
            if (!ESTADO.marcandoZona) return;
            e.preventDefault();
            elCanvasCamara.setPointerCapture(e.pointerId);
            const p = obtenerCoordenadasCamara(e);
            ESTADO.rectTemporal = { x: p.x, y: p.y, w: 0, h: 0 };
            ESTADO._origenZona = p;
        });

        elCanvasCamara.addEventListener("pointermove", (e) => {
            if (!ESTADO.marcandoZona || !ESTADO.rectTemporal || !ESTADO._origenZona) return;
            e.preventDefault();
            ESTADO.rectTemporal = rectEntre(ESTADO._origenZona, obtenerCoordenadasCamara(e));
        });

        const soltarZona = (e) => {
            if (!ESTADO.marcandoZona) return;
            e.preventDefault();
            ESTADO.marcandoZona = false;
            const r = ESTADO.rectTemporal;
            ESTADO._origenZona = null;

            if (!r || r.w < 20 || r.h < 20) {
                ESTADO.rectTemporal = null;
                mostrarAvisoCamara("Zona demasiado pequeña. Repite el arrastre.", true);
                return;
            }

            ESTADO.calibracion = ESTADO.calibracion || {};
            ESTADO.calibracion.roi = redondearRectangulo(r);
            ESTADO.rectTemporal = null;
            // Cambiar la zona invalida la referencia anterior (era de otra zona)
            ESTADO.referencia = null;
            guardarCalibracion();
            ocultarAvisoCamara();
            mostrarToast("Zona de pizarra marcada", "ok");
        };
        elCanvasCamara.addEventListener("pointerup", soltarZona);
        elCanvasCamara.addEventListener("pointercancel", soltarZona);
    }

    /* Dos puntos -> rectángulo {x, y, w, h} */
    function rectEntre(a, b) {
        const x = Math.min(a.x, b.x);
        const y = Math.min(a.y, b.y);
        return { x, y, w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
    }

    function redondearRectangulo(r) {
        return {
            x: Math.round(r.x),
            y: Math.round(r.y),
            w: Math.round(r.w),
            h: Math.round(r.h)
        };
    }

    /* La zona por defecto si el usuario aún no ha marcado nada: casi todo el
       marco, dejando un margen para no comerse los bordes del encuadre. */
    function zonaPorDefecto() {
        const margen = 0.08;
        const w = elCanvasCamara.width;
        const h = elCanvasCamara.height;
        return {
            x: Math.round(w * margen),
            y: Math.round(h * margen),
            w: Math.round(w * (1 - 2 * margen)),
            h: Math.round(h * (1 - 2 * margen))
        };
    }

    /* Captura la pizarra vacía como referencia, promediando varios fotogramas
       para eliminar ruido del sensor. */
    async function capturarReferencia() {
        if (ESTADO.modoEntrada !== "pizarron" || !ESTADO.camaraLista) return;
        if (ESTADO.capturandoReferencia) return;

        ESTADO.capturandoReferencia = true;
        try {
            /* CLAVE: usar EXACTAMENTE la misma zona y las mismas dimensiones
               que usa analizarCamara. Antes se muestreaba a un cuadrado
               lado×lado, pero el análisis lo hace respetando el aspecto
               (p. ej. 160x120). Los tamaños no cuadraban y analizarCamara
               descartaba la referencia en silencio: por eso no reconocía nada. */
            const roi = (ESTADO.calibracion && ESTADO.calibracion.roi) || zonaPorDefecto();
            const { W, H } = dimsMuestreo(roi);

            const lienzoTmp = document.createElement("canvas");
            lienzoTmp.width = W;
            lienzoTmp.height = H;
            const ctxTmp = lienzoTmp.getContext("2d", { willReadFrequently: true });

            const suma = new Float64Array(W * H);

            for (let i = 0; i < CAMARA.framesReferencia; i++) {
                mostrarAvisoCamara(
                    `Capturando pizarra vacía… deja la pizarra despejada (${i + 1}/${CAMARA.framesReferencia})`,
                    false
                );
                ctxTmp.drawImage(elCanvasCamara, roi.x, roi.y, roi.w, roi.h, 0, 0, W, H);
                const gris = aEscalaDeGrises(ctxTmp.getImageData(0, 0, W, H).data);
                for (let j = 0; j < suma.length; j++) suma[j] += gris[j];
                await esperar(70);
            }

            const referencia = new Uint8ClampedArray(suma.length);
            for (let j = 0; j < referencia.length; j++) {
                referencia[j] = Math.round(suma[j] / CAMARA.framesReferencia);
            }

            ESTADO.calibracion = ESTADO.calibracion || {};
            ESTADO.calibracion.roi = roi;
            ESTADO.calibracion.lado = CAMARA.ladoAnalisis;
            ESTADO.referencia = referencia;
            ESTADO.firmaPredicha = null; // la referencia cambió: fuerza redetectar
            guardarCalibracion();
            ocultarAvisoCamara();
            mostrarToast("Pizarra vacía capturada como referencia", "ok");
        } finally {
            ESTADO.capturandoReferencia = false;
        }
    }

    /* Convierte RGBA a un canal de gris (0-255) */
    function aEscalaDeGrises(datos) {
        const n = datos.length / 4;
        const gris = new Uint8ClampedArray(n);
        for (let i = 0, j = 0; i < datos.length; i += 4, j++) {
            gris[j] = (datos[i] * 0.299 + datos[i + 1] * 0.587 + datos[i + 2] * 0.114) | 0;
        }
        return gris;
    }

    function esperar(ms) {
        return new Promise((res) => setTimeout(res, ms));
    }

    /* ----- Persistencia de la calibración en localStorage ----- */

    function guardarCalibracion() {
        try {
            const referencia = ESTADO.referencia ? bytesABase64(ESTADO.referencia) : null;
            const datos = {
                version: 1,
                roi: ESTADO.calibracion ? ESTADO.calibracion.roi : null,
                lado: CAMARA.ladoAnalisis,
                referencia
            };
            localStorage.setItem(CLAVE_LS_CALIBRACION, JSON.stringify(datos));
        } catch (e) {
            console.warn("[AdivinIA] No se pudo guardar la calibración:", e.message);
        }
    }

    function cargarCalibracion() {
        try {
            const crudo = localStorage.getItem(CLAVE_LS_CALIBRACION);
            if (!crudo) return;
            const datos = JSON.parse(crudo);
            if (datos && datos.roi) {
                ESTADO.calibracion = { roi: datos.roi, lado: datos.lado || CAMARA.ladoAnalisis };
            }
            if (datos && datos.referencia) {
                ESTADO.referencia = base64ABytes(datos.referencia);
            }
        } catch (e) {
            console.warn("[AdivinIA] No se pudo leer la calibración guardada:", e.message);
        }
    }

    function bytesABase64(bytes) {
        let binario = "";
        const trozo = 0x8000; // evita desbordar la pila con arrays grandes
        for (let i = 0; i < bytes.length; i += trozo) {
            binario += String.fromCharCode.apply(null, bytes.subarray(i, i + trozo));
        }
        return btoa(binario);
    }

    function base64ABytes(b64) {
        const binario = atob(b64);
        const bytes = new Uint8ClampedArray(binario.length);
        for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);
        return bytes;
    }

    /* ======================================================================
       12. MÁSCARA DE TINTA (pincel virtual)
       ======================================================================
       La cámara no "dibuja": compara cada frame con la pizarra vacía guardada
       y marca como tinta todo píxel claramente MÁS OSCURO que la referencia.
       El resultado es una imagen blanco/negro que entra al mismo modelo que
       usa el modo ratón, a través de normalizarLienzo(). */

    /* Canvas fuera de pantalla donde se muestrea el ROI (a tamaño pequeño) */
    const canvasMuestra = document.createElement("canvas");
    const ctxMuestra = canvasMuestra.getContext("2d", { willReadFrequently: true });

    /* Canvas fuera de pantalla con la máscara binaria (blanco / negro) */
    const canvasMascara = document.createElement("canvas");
    const ctxMascara = canvasMascara.getContext("2d", { willReadFrequently: true });

    let idAnalisisCamara = null;

    /* Proporción del ROI que se conserva al muestrear: la imagen pequeña
       mantiene la relación de aspecto para no deformar el dibujo. */
    function dimsMuestreo(roi) {
        const lado = CAMARA.ladoAnalisis;
        const k = lado / Math.max(roi.w, roi.h);
        return {
            W: Math.max(1, Math.round(roi.w * k)),
            H: Math.max(1, Math.round(roi.h * k))
        };
    }

    function ajustarCanvas(canvas, w, h) {
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
    }

    /* Muestrea el ROI como escala de grises a tamaño reducido */
    function muestrearRoiGris(roi, W, H) {
        ajustarCanvas(canvasMuestra, W, H);
        ctxMuestra.imageSmoothingEnabled = true;
        ctxMuestra.imageSmoothingQuality = "high";
        ctxMuestra.drawImage(
            elCanvasCamara,
            roi.x, roi.y, roi.w, roi.h,
            0, 0, W, H
        );
        const datos = ctxMuestra.getImageData(0, 0, W, H).data;
        return aEscalaDeGrises(datos);
    }

    /* Devuelve un Uint8Array (1 = tinta) comparando contra la referencia.
       Luego erosiona para quitar puntitos sueltos de ruido. */
    function construirMascara(gris, ref, W, H) {
        const umbral = CAMARA.umbralTinta;
        const n = W * H;
        const esTinta = new Uint8Array(n);

        for (let i = 0; i < n; i++) {
            // Tinta = claramente más oscuro que la pizarra vacía
            esTinta[i] = (ref[i] - gris[i]) > umbral ? 1 : 0;
        }

        const min = CAMARA.vecinosMinimos;
        if (min > 0) {
            const copia = esTinta.slice();
            for (let y = 0; y < H; y++) {
                for (let x = 0; x < W; x++) {
                    const idx = y * W + x;
                    if (!copia[idx]) continue;
                    let vecinos = 0;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            if (dx === 0 && dy === 0) continue;
                            const nx = x + dx;
                            const ny = y + dy;
                            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                            if (copia[ny * W + nx]) vecinos++;
                        }
                    }
                    if (vecinos < min) esTinta[idx] = 0;
                }
            }
        }

        /* Dilatación: engorda el trazo para acercarlo al grosor de Quick, Draw!
           Un rotulador fino, reducido a 160 px, queda como una línea de 1-2 px
           que al bajar a 28 px casi desaparece. Engrosarlo hace que el modelo
           lo reconozca igual que un trazo de ratón. */
        for (let p = 0; p < CAMARA.dilatacion; p++) {
            const copia = esTinta.slice();
            for (let y = 0; y < H; y++) {
                for (let x = 0; x < W; x++) {
                    if (copia[y * W + x]) continue;
                    if ((x > 0 && copia[y * W + x - 1]) ||
                        (x < W - 1 && copia[y * W + x + 1]) ||
                        (y > 0 && copia[(y - 1) * W + x]) ||
                        (y < H - 1 && copia[(y + 1) * W + x])) {
                        esTinta[y * W + x] = 1;
                    }
                }
            }
        }

        return esTinta;
    }

    /* Vuelca la máscara al canvas y devuelve la fracción de tinta (0..1).
       Dilata ligeramente el trazo (grosor) para acercarlo al de Quick, Draw!,
       cuyo trazo es más ancho que el de un rotulador fino. */
    function pintarMascara(esTinta, W, H) {
        const img = ctxMascara.createImageData(W, H);
        const datos = img.data;
        let tinta = 0;
        for (let i = 0; i < W * H; i++) {
            const v = esTinta[i] ? 0 : 255;
            if (esTinta[i]) tinta++;
            const p = i * 4;
            datos[p] = v;
            datos[p + 1] = v;
            datos[p + 2] = v;
            datos[p + 3] = 255;
        }
        ctxMascara.putImageData(img, 0, 0);
        return tinta / (W * H);
    }

    function dibujarPreviewProcesado() {
        if (!ctxProcesado) return;
        ctxProcesado.imageSmoothingEnabled = false;
        ctxProcesado.fillStyle = "#ffffff";
        ctxProcesado.fillRect(0, 0, elCanvasProcesado.width, elCanvasProcesado.height);
        if (canvasMascara.width && canvasMascara.height) {
            ctxProcesado.drawImage(
                canvasMascara,
                0, 0, canvasMascara.width, canvasMascara.height,
                0, 0, elCanvasProcesado.width, elCanvasProcesado.height
            );
        }
    }

    /* Huella COARSE de una máscara: sirve para saber si el dibujo cambió sin
       guardar la imagen entera. Se cuantiza (bbox a 5% y fracción en %) porque
       un hash exacto cambiaría en cada frame por el simple temblor de la
       cámara, y la histéresis nunca llegaría a considerarlo "estable". */
    function firmaMascara(esTinta, W, H) {
        let minX = W, minY = H, maxX = -1, maxY = -1, cuenta = 0;
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                if (!esTinta[y * W + x]) continue;
                cuenta++;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        if (cuenta === 0) return "0";

        const col = (v) => Math.round((v / W) * 20); // 20 franjas horizontales
        const fil = (v) => Math.round((v / H) * 20); // 20 franjas verticales
        const porc = Math.round((cuenta / (W * H)) * 100);
        return [col(minX), fil(minY), col(maxX), fil(maxY), porc].join(":");
    }

    /* Un tick de análisis: máscara + predicción si hay suficiente tinta */
    function analizarCamara() {
        if (ESTADO.modoEntrada !== "pizarron" || !ESTADO.camaraLista) return;
        if (!ESTADO.calibracion || !ESTADO.calibracion.roi || !ESTADO.referencia) return;

        const roi = ESTADO.calibracion.roi;
        const { W, H } = dimsMuestreo(roi);

        // La referencia guardada debe ser de esta misma zona y tamaño
        if (ESTADO.referencia.length !== W * H) {
            console.warn("[AdivinIA] Referencia de tamaño incorrecto (" +
                ESTADO.referencia.length + " vs " + (W * H) + "). Se descarta: recalibra.");
            ESTADO.referencia = null;
            guardarCalibracion(); // limpia también la copia guardada, para no repetir el aviso
            mostrarAvisoCamara("Calibración caducada. Pulsa CAPTURAR PIZARRA VACÍA.", false);
            return;
        }

        const gris = muestrearRoiGris(roi, W, H);
        const esTinta = construirMascara(gris, ESTADO.referencia, W, H);

        ajustarCanvas(canvasMascara, W, H);
        const fraccion = pintarMascara(esTinta, W, H);
        dibujarPreviewProcesado();

        // Pizarra prácticamente vacía: no molestamos al modelo ni a la voz
        if (fraccion < CAMARA.umbralMinTinta) {
            ESTADO.firmaMascara = null;
            ESTADO.firmaEstable = 0;
            ESTADO.firmaPredicha = null;
            return;
        }

        /* Histéresis: solo predecimos cuando el dibujo lleva varias capturas
           sin cambiar. Si no, cada frame dispararía una predicción distinta y
           las barras y la voz temblarían sin parar. */
        const firma = firmaMascara(esTinta, W, H);
        if (firma === ESTADO.firmaMascara) {
            ESTADO.firmaEstable++;
        } else {
            ESTADO.firmaMascara = firma;
            ESTADO.firmaEstable = 1;
            return;
        }

        if (ESTADO.firmaEstable < CAMARA.framesEstables) return;
        if (firma === ESTADO.firmaPredicha) return; // ya se predijo esta forma

        ESTADO.firmaPredicha = firma;

        const valores = normalizarLienzo(canvasMascara);
        if (!valores) return;

        ejecutarPipeline(valores);
    }

    function arrancarAnalisisCamara() {
        pararAnalisisCamara();
        idAnalisisCamara = setInterval(analizarCamara, Math.round(1000 / CAMARA.fpsAnalisis));
    }

    function pararAnalisisCamara() {
        if (idAnalisisCamara !== null) {
            clearInterval(idAnalisisCamara);
            idAnalisisCamara = null;
        }
    }

    /* ----- Modo de entrada ----- */
    function guardarModo(modo) {
        try {
            localStorage.setItem(CLAVE_LS_MODO, modo);
        } catch (e) {
            console.warn("[AdivinIA] No se pudo guardar el modo:", e.message);
        }
    }

    function leerModoGuardado() {
        try {
            const m = localStorage.getItem(CLAVE_LS_MODO);
            if (m === "pizarron" || m === "raton") return m;
        } catch (e) {
            console.warn("[AdivinIA] No se pudo leer el modo guardado:", e.message);
        }
        return MODO_ENTRADA_POR_DEFECTO;
    }

    async function cambiarModoEntrada(modo) {
        if (modo !== "pizarron" && modo !== "raton") modo = "raton";
        ESTADO.modoEntrada = modo;
        guardarModo(modo);
        aplicarModoEntrada();

        if (modo === "pizarron") {
            try {
                if (!ESTADO.camaraLista) await iniciarCamara();
                if (!ESTADO.referencia) {
                    // Falta el paso clave: sin referencia no hay detección posible
                    mostrarAvisoCamara(
                        "Deja la pizarra vacía y pulsa CAPTURAR PIZARRA VACÍA para empezar.",
                        false
                    );
                    actualizarEstado("Modo pizarrón: falta capturar la pizarra vacía.", "");
                } else {
                    actualizarEstado("Modo pizarrón: cámara activa. Dibuja en la pizarra.", "listo");
                }
            } catch (e) {
                const msg = manejarErrorCamara(e);
                console.error("[AdivinIA] No se pudo iniciar la cámara:", msg, e);
                mostrarToast("Sin cámara: paso a modo RATÓN", "error");
                // Fallback: nunca dejamos la pantalla bloqueada por la cámara
                await cambiarModoEntrada("raton");
            }
        } else {
            detenerCamara();
            actualizarEstado("Modo ratón. Dibuja sobre la pantalla.", "");
        }
    }

    function aplicarModoEntrada() {
        const esPizarron = ESTADO.modoEntrada === "pizarron";

        if (elContenedorCamara) elContenedorCamara.hidden = !esPizarron;
        if (elContenedorLienzo) elContenedorLienzo.hidden = esPizarron;

        if (elBtnModoPizarron) {
            elBtnModoPizarron.classList.toggle("activo", esPizarron);
            elBtnModoPizarron.setAttribute("aria-pressed", String(esPizarron));
        }
        if (elBtnModoRaton) {
            elBtnModoRaton.classList.toggle("activo", !esPizarron);
            elBtnModoRaton.setAttribute("aria-pressed", String(!esPizarron));
        }

        // Los controles de calibración solo tienen sentido en pizarrón
        [elBtnCalibrarZona, elBtnCalibrarFondo, elBtnCambiarCamara].forEach((b) => {
            if (b) b.hidden = !esPizarron;
        });
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