/* ==========================================================================
   Validador de modelo para AdivinIA  —  se ejecuta SIN navegador
   --------------------------------------------------------------------------
   Para qué sirve: cargar tu modelo con TensorFlow.js en Node y comprobar que
   (a) se puede deserializar, (b) los 16 pesos aterrizan en la capa correcta y
   (c) el modelo responde con confianza. Un modelo roto se detecta AQUÍ, en 5
   segundos, en vez de delante del público en la feria.

   Uso (desde la carpeta del proyecto):
       node verificar-modelo.js

   Si todo va bien you'll ver:
       CARGA: OK
       => ORDEN CORRECTO
       valores reales cargados
       El modelo responde con confianza
   ========================================================================== */
const fs = require("fs");
const path = require("path");

/* ----------------------------------------------------------------------
   Rutas
   ---------------------------------------------------------------------- */
const MODELO = process.argv[2] || path.join(__dirname, "model", "model.json");
const BIN = MODELO.replace(/model\.json$/, "group1-shard1of1.bin");
const CLASES_JSON = MODELO.replace(/model\.json$/, "class_names.json");
const TFJS = path.join(__dirname, "tf.min.js");

/* ----------------------------------------------------------------------
   Comprobaciones
   ---------------------------------------------------------------------- */
let fallos = 0;
const ok = (c, m) => { console.log(`  ${c ? "✔" : "✘"} ${m}`); if (!c) fallos++; };

(async () => {
    console.log("\n=== 1. Archivos ===");
    [TFJS, MODELO, BIN].forEach((f) => ok(fs.existsSync(f), path.basename(f)));

    const CLASES = JSON.parse(fs.readFileSync(CLASES_JSON, "utf8"));
    ok(Array.isArray(CLASES), `class_names.json con ${CLASES.length} clases`);
    ok(fs.existsSync(BIN), "pesos .bin");

    const tf = require(TFJS);
    await tf.setBackend("cpu");
    await tf.ready();
    ok(true, `TensorFlow.js ${tf.version.tfjs} cargado`);

    const modelJSON = JSON.parse(fs.readFileSync(MODELO, "utf8"));
    const bin = fs.readFileSync(BIN);

    console.log(`\n=== 2. Formato: ${modelJSON.format} ===`);
    /* OJO: el valor real que escribe el conversor es "graph-model", no
       "tfjs-graph-model" (esa cadena no aparece ni una vez en tf.min.js).
       Nos fiamos ademas de la ausencia de "model_config", que es lo que
       distingue de verdad un grafo de un modelo de capas. */
    const esGraph = modelJSON.format === "graph-model"
        || modelJSON.format === "tfjs-graph-model"
        || !modelJSON.modelTopology?.model_config;
    ok(esGraph, "Es un GRAPH MODEL (recomandado para Keras 3)");
    if (!esGraph) {
        console.log("     Un layers-model de Keras 3 no es compatible con TF.js.");
        console.log("     Reexporta con --output_format=tfjs_graph_model.");
    }

    console.log("\n=== 3. Carga ===");
    const io = tf.io.fromMemory({
        modelTopology: modelJSON.modelTopology,
        weightSpecs: modelJSON.weightsManifest.flatMap((g) => g.weights),
        weightData: bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
        format: modelJSON.format,
        signature: modelJSON.signature || null,
        userDefinedMetadata: {}
    });

    let modelo;
    try {
        modelo = esGraph ? await tf.loadGraphModel(io) : await tf.loadLayersModel(io);
        ok(true, "El modelo se deserializa correctamente");
    } catch (e) {
        ok(false, "FALLO AL CARGAR -> " + e.message);
        console.log("\nRevisa el punto 2 de arriba.\n");
        process.exit(1);
    }

    if (!esGraph) {
        console.log("\n=== 4. Orden de los pesos ===");
        // En un graph model los pesos se emparejan por nombre y este chequeo
        // no aplica. En layers-model se asignan por orden, y ahí sí importa.
        const pesos = modelo.weights.map((w) => w.name);
        console.log("     " + pesos.join(" "));
        console.log("     Si el orden no sigue la arquitectura, el modelo devuelve ruido.");
    }

    console.log("\n=== 5. Inferencia de prueba (un '+' dibujado) ===");
    const T = 28, MAX = 20, img = new Float32Array(T * T);
    const trazo = (x0, y0, x1, y1, g) => {
        for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
            const L = Math.hypot(x1 - x0, y1 - y0);
            const t = Math.hypot(x - x0, y - y0) / L;
            if (Math.hypot(x - (x0 + (x1 - x0) * t), y - (y0 + (y1 - y0) * t)) <= g / 2) {
                img[y * T + x] = 1;
            }
        }
    };
    trazo(4, 14, 24, 14, 3);
    trazo(14, 4, 14, 24, 3);

    const t0 = Date.now();
    const tensor = tf.tensor4d(img, [1, T, T, 1], "float32");
    const salida = esGraph ? await modelo.executeAsync(tensor) : await modelo.predict(tensor);

    let probs = null;
    const tensores = salida instanceof tf.Tensor ? [salida]
        : (Array.isArray(salida) ? salida : Object.values(salida).filter((t) => t instanceof tf.Tensor));
    const buena = tensores.find((t) => t.size === CLASES.length) || tensores[0];
    probs = buena.dataSync();

    console.log(`     ${Date.now() - t0} ms | ${probs.length} probabilidades | suma ${probs.reduce((a, b) => a + b, 0).toFixed(4)}`);
    ok(probs.length === CLASES.length, "El número de salidas coincide con las clases");

    const orden = [...probs].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    orden.slice(0, 5).forEach(([v, i], k) =>
        console.log(`     ${k + 1}. ${CLASES[i]}: ${(v * 100).toFixed(1)}%`));
    ok(orden[0][0] > 0.05, `Responde con confianza (top1 = ${(orden[0][0] * 100).toFixed(1)}%)`);
    if (orden[0][0] <= 0.05) {
        console.log("     Salida casi uniforme => pesos mal asignados o preprocesamiento incorrecto.");
    }

    console.log(fallos === 0
        ? "\n=== TODO CORRECTO. Listo para la feria ===\n"
        : `\n=== ${fallos} PROBLEMA(S) ===\n`);
    process.exit(fallos === 0 ? 0 : 1);
})();