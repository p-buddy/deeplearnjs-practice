/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


var dl = deeplearn;
var AdadeltaOptimiz = dl.AdadeltaOptimiz;
var AdagradOptimize = dl.AdagradOptimize;
var AdamaxOptimizer = dl.AdamaxOptimizer;
var AdamOptimizer = dl.AdamOptimizer;
var Array1D = dl.Array1D;
var Array3D = dl.Array3D;
var DataStats = dl.DataStats;
var FeedEntry = dl.FeedEntry;
var Graph = dl.Graph;
// var GraphRunner = dl.GraphRunner;
var InCPUMemoryShuf = dl.InCPUMemoryShuf;
var InMemoryDataset = dl.InMemoryDataset;
var MetricReduction = dl.MetricReduction;
var MomentumOptimiz = dl.MomentumOptimiz;
var NDArray = dl.NDArray;
var NDArrayMath = dl.NDArrayMath;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Optimizer = dl.Optimizer;
var RMSPropOptimize = dl.RMSPropOptimize;
var Scalar = dl.Scalar;
var Session = dl.Session;
var SGDOptimizer = dl.SGDOptimizer;
var MomentumOptimizer = dl.MomentumOptimizer;
var Tensor = dl.Tensor;
var util = dl.util;
var xhr_dataset = dl.xhr_dataset;
var XhrDataset = dl.XhrDataset;
var XhrDatasetConfi = dl.XhrDatasetConfi;

var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;


function getDisplayShape(shape) {
    return `[${shape}]`;
}

const DATASETS_CONFIG_JSON = 'deeplearnjs/mnist/model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 1;
const INFERENCE_IMAGE_SIZE_PX = 100;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

var Normalization = {
    NORMALIZATION_NEGATIVE_ONE_TO_ONE: 0,
    NORMALIZATION_ZERO_TO_ONE: 1,
    NORMALIZATION_NONE: 2
}

var selectedNormalizationOption = Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE;


var ApplicationState = {
    IDLE: 1,
    TRAINING: 2
}


var isValid = null;
// totalTimeSec: string;
// applicationState: ApplicationState;
var modelInitialized = false; //: boolean;
// showTrainStats: boolean;
// selectedNormalizationOption: number;

// // Datasets and models.
// graphRunner: GraphRunner;
var graph; //: Graph;
var graphRunner = null;
var session = null;
var optimizer; //: Optimizer;
var xTensor; //: Tensor;
var labelTensor; //: Tensor;
var costTensor; //: Tensor;
var accuracyTensor; //: Tensor;
var predictionTensor; //: Tensor;

// datasetDownloaded: boolean;
// datasetNames: string[];
var selectedDatasetName;
var modelNames;
// selectedModelName: string;
var optimizerNames; //: string[];
var selectedOptimizerName;
var selectedEnvName;
var loadedWeights;
var dataSets;
//: {
//     [datasetName: string]: InMemoryDataset
// };
var dataSet;
var xhrDatasetConfigs;
//: {
//     [datasetName: string]: XhrDatasetConfig
// };
// datasetStats: DataStats[];
var learningRate; //: number;
var momentum; //: number;
var needMomentum; //: boolean;
var gamma; //: number;
var needGamma; //: boolean;
var beta1; //: number;
var needBeta1; //: boolean;
var beta2; //: number;
var needBeta2; //: boolean;
var batchSize; //: number;

// // Stats.
// showDatasetStats: boolean;
// statsInputRange: string;
// statsInputShapeDisplay: string;
// statsLabelShapeDisplay: string;
// statsExampleCount: number;

// trainButton: HTMLButtonElement;

// // Visualizers.
var inputNDArrayVisualizers;
var outputNDArrayVisualizers; //: NDArrayLogitsVisualizer[];

var inputShape;
var labelShape;
var examplesPerSec; //: number;
// examplesTrained: number;
var inferencesPerSec; //: number;
// inferenceDuration: number;

// inputLayer: ModelLayer;
var hiddenLayers = [];

var layersContainer;

var math; //: NDArrayMath;
// // Keep one instance of each NDArrayMath so we don't create a user-initiated
// // number of NDArrayMathGPU's.
var mathGPU; //: NDArrayMathGPU;
var mathCPU; //: NDArrayMathCPU;

var dl = deeplearn;
var Array1D = dl.Array1D;
var Array2D = dl.Array2D;
var Array4D = dl.Array4D;



function load_weights() {
    var varLoader = new dl.CheckpointLoader('.');

    varLoader.getAllVariables().then(async vars => {
      const conv1B = vars['conv1/biases'];
      const conv1W = vars['conv1/weights'];
      const conv2W = vars['conv2/weights'];

      console.log(conv1B);
      console.log(conv1W);
      console.log(conv2W);

    });
}


function getTestData() {
    const data = dataSet.getData();
    if (data == null) {
        return null;
    }
    const [images, labels] = dataSet.getData();

    var toprint;
    for (var i = 0; i < images.length; i++) {
      toprint = images[i][0]
    }

    console.log(toprint);

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);


    return [images.slice(start), labels.slice(start)];
}

function getRandomTest() {
  const data = dataSet.getData();
}

function startInference() {

    console.log('starting inference...');

    const testData = getTestData();
    if (testData == null) {
        // Dataset not ready yet.
        return;
    }
    if (isValid && (testData != null)) {
        // console.log(testData);
        const inferenceShuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder(testData);
        const [inferenceInputProvider, inferenceLabelProvider] =
        inferenceShuffledInputProviderGenerator.getInputProviders();

        const inferenceFeeds = [{
                tensor: xTensor,
                data: inferenceInputProvider
            },
            {
                tensor: labelTensor,
                data: inferenceLabelProvider
            }
        ];

        console.log('inferring...');

        graphRunner.infer(
            predictionTensor, inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS,
            INFERENCE_EXAMPLE_COUNT);
    }
}

function createModel() {

    /*
    TO BE customized
    */

    if (session != null) {
        session.dispose();
    }

    modelInitialized = false;
    if (isValid === false) {
        return;
    }

    console.log('creating model...');

    graph = new Graph();
    const g = graph;
    xTensor = g.placeholder('input', inputShape);
    labelTensor = g.placeholder('label', labelShape);

    let network = xTensor;

    for (let i = 0; i < hiddenLayers.length; i++) {
        let weights = null;
        if (loadedWeights != null) {
            weights = loadedWeights[i];
        }
        network = hiddenLayers[i].addLayer(g, network, i, weights);
    }
    predictionTensor = network;
    costTensor =
        g.softmaxCrossEntropyCost(predictionTensor, labelTensor);
    accuracyTensor =
        g.argmaxEquals(predictionTensor, labelTensor);

    var loadedWeights = null;

    var session = new Session(g, math);
    graphRunner.setSession(session);

    // startInference();

    modelInitialized = true;
}

function removeAllLayerTableRow(elt) {
    while (elt.hasChildNodes()) {
        elt.removeChild(elt.lastChild);
    }
}

function insertLayerTableRow(elt, name, inShape, outShape) {

    elt.classList.add('table-responsive');
    elt.style.border = 'none';

    var h = document.createElement("h5");
    h.appendChild(document.createTextNode(`${name}`));

    var table = document.createElement('table');
    // table.classList.add('table');
    table.style.width = "100px";

    var head = table.createTHead();
    var row = head.insertRow(0);

    row.insertCell(0).outerHTML = `<th>in</th>`;
    row.insertCell(1).outerHTML = `<th>out</th>`;

    var body = table.createTBody();
    var row = body.insertRow(0);

    row.insertCell(0).innerHTML = `${inShape}`;
    row.insertCell(1).innerHTML = `${outShape}`;

    elt.appendChild(h);
    elt.appendChild(table);
}

function updateSelectedDataset(datasetName) {
    if (dataSet != null) {
        dataSet.removeNormalization(IMAGE_DATA_INDEX);
    }

    graphRunner.stopInferring();

    if (dataSet != null) {
        dataSet.dispose();
    }

    selectedDatasetName = datasetName;
    var selectedModelName = '';
    dataSet = dataSets[datasetName];
    var datasetDownloaded = false;
    var showDatasetStats = false;

    dataSet.fetchData().then(() => {
        datasetDownloaded = true;
        applyNormalization(selectedNormalizationOption);
        if (isValid) {
            console.log('selected dataset');
            createModel();
            // startInference();
        }
        // Get prebuilt models.
        populateModelDropdown();
    });

    inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
    labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);

    layersContainer = document.querySelector('#hidden-layers');

    var inputLayer = document.querySelector('#input-layer');
    removeAllLayerTableRow(inputLayer);
    insertLayerTableRow(inputLayer, 'input-layer', null, getDisplayShape(inputShape));

    labelShapeDisplay = getDisplayShape(labelShape);

    const costLayer = document.querySelector('#cost-layer');
    removeAllLayerTableRow(costLayer);
    insertLayerTableRow(costLayer, 'cost-layer', labelShapeDisplay, labelShapeDisplay);

    const outputLayer = document.querySelector('#output-layer');
    removeAllLayerTableRow(outputLayer);
    insertLayerTableRow(outputLayer, 'output-layer', labelShapeDisplay, null);

    // Setup the inference example container.
    // TODO(nsthorat): Generalize
    const inferenceContainer =
        document.querySelector('#inference-container');
    inferenceContainer.innerHTML = '';
    inputNDArrayVisualizers = [];
    outputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';

        // Set up the input visualizer.

        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement);
        ndarrayImageVisualizer.setShape(this.inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);

        inputNDArrayVisualizers.push(ndarrayImageVisualizer);
        // inferenceExampleElement.appendChild(ndarrayImageVisualizer);

        // Set up the output ndarray visualizer.
        const ndarrayLogitsVisualizer = new NDArrayLogitsVisualizer(inferenceExampleElement, 3);
        document.createElement('ndarray-logits-visualizer');
        ndarrayLogitsVisualizer.initialize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
        // inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

        inferenceContainer.appendChild(inferenceExampleElement);
    }
}



function populateDatasets() {
    dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(_xhrDatasetConfigs => {
                for (const datasetName in _xhrDatasetConfigs) {
                    if (_xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                        dataSets[datasetName] =
                            new XhrDataset(_xhrDatasetConfigs[datasetName]);
                    }
                }
                var datasetNames = Object.keys(dataSets);
                selectedDatasetName = datasetNames[0];
                xhrDatasetConfigs = _xhrDatasetConfigs;
                updateSelectedDataset(selectedDatasetName);
            },
            error => {
                throw new Error(`Dataset config could not be loaded: ${error}`);
            });
}


function applyNormalization(selectedNormalizationOption) {
    switch (selectedNormalizationOption) {
        case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                break;
            }
        case Normalization.NORMALIZATION_ZERO_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
                break;
            }
        case Normalization.NORMALIZATION_NONE:
            {
                dataSet.removeNormalization(IMAGE_DATA_INDEX);
                break;
            }
        default:
            {
                throw new Error('Normalization option must be 0, 1, or 2');
            }
    }

}


function populateModelDropdown() {
    const _modelNames = ['Custom'];

    const modelConfigs =
        xhrDatasetConfigs[selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
        if (modelConfigs.hasOwnProperty(modelName)) {
            _modelNames.push(modelName);
        }
    }
    modelNames = _modelNames;
    selectedModelName = _modelNames[_modelNames.length - 1];
    updateSelectedModel(selectedModelName);

    var modelDropdown = document.getElementById("model-dropdown");

    for (let i = 0; i < modelDropdown.options.length; ++i) {
        if (modelDropdown.options[i].text === selectedModelName)
            modelDropdown.options[i].selected = true;
    }

}

function validateModel() {
    let valid = true;
    for (let i = 0; i < hiddenLayers.length; ++i) {
        valid = valid && hiddenLayers[i].isValid();
    }
    if (hiddenLayers.length > 0) {
        const lastLayer = hiddenLayers[hiddenLayers.length - 1];
        valid = valid &&
            util.arraysEqual(labelShape, lastLayer.getOutputShape());
    }
    isValid = valid && (hiddenLayers.length > 0);
}

function layerParamChanged() {
    // Go through each of the model layers and propagate shapes.
    let lastOutputShape = inputShape;
    for (let i = 0; i < hiddenLayers.length; i++) {
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);
    }
    validateModel();

    if (isValid) {
        createModel();
        // startInference();
    }
}

function removeAllLayers() {
    for (let i = 0; i < hiddenLayers.length; i++) {
        layersContainer.removeChild(hiddenLayers[i].paramContainer);
    }
    hiddenLayers = [];
    layerParamChanged();
}

function updateSelectedModel(modelName) {
    removeAllLayers();
    if (modelName === 'Custom') {
        // TODO(nsthorat): Remember the custom layers.
        return;
    }

    loadModelFromPath(xhrDatasetConfigs[selectedDatasetName].modelConfigs[modelName].path);
}


function addLayer() {
    const modelLayer = new ModelLayer(); //document.createElement('model-layer');
    // var t = document.createTextNode("hidden-layer in:" + labelShapeDisplay);
    // modelLayer.appendChild(t)
    // modelLayer.appendChild(document.createElement("br"));
    // modelLayer.className = 'model-layer';

    const lastHiddenLayer = hiddenLayers[this.hiddenLayers.length - 1];
    const lastOutputShape = lastHiddenLayer != null ?
        lastHiddenLayer.getOutputShape() :
        inputShape;
    hiddenLayers.push(modelLayer);
    modelLayer.initialize(window, lastOutputShape);

    layersContainer.appendChild(modelLayer.paramContainer);
    return modelLayer;
}

function loadModelFromJson(modelJson) {
    let lastOutputShape = inputShape;

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = addLayer();
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);
        insertLayerTableRow(modelLayer.paramContainer,
            modelLayer.selectedLayerName,
            modelLayer.inputShapeDisplay,
            modelLayer.outputShapeDisplay)
    }
    validateModel();
}

function loadModelFromPath(modelPath) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
        loadModelFromJson(xhr.responseText);
    };
    xhr.onerror = (error) => {
        throw new Error(`Model could not be fetched from ${modelPath}: ${error}`);
    };
    xhr.send();
}


function displayInferenceExamplesOutput(
    inputFeeds, inferenceOutputs) {
    let images = [];
    const logits = [];
    const labels = [];
    for (let i = 0; i < inputFeeds.length; i++) {
        images.push(inputFeeds[i][IMAGE_DATA_INDEX].data);
        labels.push(inputFeeds[i][LABEL_DATA_INDEX].data);
        logits.push(inferenceOutputs[i]);
    }

    images =
        dataSet.unnormalizeExamples(images, IMAGE_DATA_INDEX);

    // Draw the images.
    for (let i = 0; i < inputFeeds.length; i++) {
        inputNDArrayVisualizers[i].saveImageDataFromNDArray(images[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {
        const softmaxLogits = math.softmax(logits[i]);

        outputNDArrayVisualizers[i].drawLogits(
            softmaxLogits, labels[i],
            xhrDatasetConfigs[selectedDatasetName].labelClassNames);
        inputNDArrayVisualizers[i].draw();

        softmaxLogits.dispose();
    }
}

function smoothExamplesPerSec(
    lastExamplesPerSec, nextExamplesPerSec) {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
            (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
        .toPrecision(3));
}

function displayInferenceExamplesPerSec(examplesPerSec) {
    inferencesPerSec =
        smoothExamplesPerSec(inferencesPerSec, examplesPerSec);
    inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));
    document.getElementById("inferencesPerSec").innerHTML = `Inferences/sec: ${inferencesPerSec}`;
    document.getElementById("inferenceDuration").innerHTML = `inference Duration: ${inferenceDuration} ms`;
}


mathGPU = new NDArrayMathGPU();
mathCPU = new NDArrayMathCPU();

function run() {
    // math = mathGPU;
    var envDropdown = document.getElementById("environment-dropdown");
    selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName, graphRunner);

    const eventObserver = {
        batchesTrainedCallback: (batchesTrained) =>
            displayBatchesTrained(batchesTrained),
        avgCostCallback: (avgCost) => displayCost(avgCost),
        metricCallback: (metric) => displayAccuracy(metric),
        inferenceExamplesCallback:
            (inputFeeds, inferenceOutputs) =>
            displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),
        inferenceExamplesPerSecCallback: (examplesPerSec) =>
            displayInferenceExamplesPerSec(examplesPerSec),
        trainExamplesPerSecCallback: (examplesPerSec) =>
            displayExamplesPerSec(examplesPerSec),
        totalTimeCallback: (totalTimeSec) => {
            totalTimeSec = totalTimeSec.toFixed(1);
            document.getElementById("totalTimeSec").innerHTML = `Total time: ${totalTimeSec} sec.`;
        },
    };
    graphRunner = new GraphRunner(math, session, eventObserver);

    // Set up datasets.
    populateDatasets();

    document.querySelector('#dataset-dropdown').addEventListener(
        'change', (event) => {

            if (graphRunner != null) {
                     graphRunner = null;
            }
            graphRunner = new GraphRunner(math, session, eventObserver);

            // Update the dataset.
            const datasetName = event.target.value;
            updateSelectedDataset(datasetName);

            // TODO(nsthorat): Remember the last model used for each dataset.
            removeAllLayers();
            console.log('dataset =', datasetName)
        });
    document.querySelector('#model-dropdown').addEventListener(
        'change', (event) => {

            if (graphRunner != null) {
                     graphRunner = null;
            }
            graphRunner = new GraphRunner(math, session, eventObserver);

            // Update the model.

            const modelName = event.target.value;
            updateSelectedModel(modelName);
            console.log('model =', modelName)
        });


    applicationState = ApplicationState.IDLE;
    // var loadedWeights = null;
    // var modelInitialized = false;
    // var showTrainStats = false;
    // var showDatasetStats = false;

    // const addButton = document.querySelector('#add-layer');
    // addButton.addEventListener('click', () => addLayer());

    // const downloadModelButton = document.querySelector('#download-model');
    // downloadModelButton.addEventListener('click', () => downloadModel());
    // const uploadModelButton = document.querySelector('#upload-model');
    // uploadModelButton.addEventListener('click', () => uploadModel());
    // setupUploadModelButton();

    // const uploadWeightsButton = document.querySelector('#upload-weights');
    // uploadWeightsButton.addEventListener('click', () => uploadWeights());
    // setupUploadWeightsButton();

    document.querySelector('#environment-dropdown').addEventListener('change', (event) => {
        selectedEnvName = event.target.value;
        updateSelectedEnvironment(selectedEnvName, graphRunner)
    });

    hiddenLayers = [];
    examplesPerSec = 0;
    inferencesPerSec = 0;
}

function updateSelectedEnvironment(selectedEnvName, _graphRunner = null) {

    math = (selectedEnvName === 'GPU') ? mathGPU : mathCPU;
    console.log('math =', math == mathGPU ? 'mathGPU' : 'mathCPU')
    if (_graphRunner != null) {
        _graphRunner.setMath(math);
    }

}

var infer_request = null;
var btn_infer = document.getElementById('buttoninfer');
var infer_paused = true;
btn_infer.addEventListener('click', () => {

    infer_paused = !infer_paused;
    if (infer_paused) {
        btn_infer.value = 'Start Inferring';
        if (graphRunner != null) {
            graphRunner.stopInferring();
        }


    } else {

        infer_request = true;
        btn_infer.value = 'Pause Inferring';


    }
});




var train_request = null;
var train_paused = true;

var updateNetParamDisplay = function () {
    var z = 1;
    // document.getElementById('decay_input').value = trainer.l2_decay;
}

function monitor() {

    if (modelInitialized == false) {

        btn_infer.disabled = true;
        btn_infer.value = 'Initializing Model ...'
        // btn_train.disabled = true;

    } else {
        if (isValid) {

            btn_infer.disabled = false;

            if (infer_paused) {
                btn_infer.value = 'Start Infering'
            } else {
                btn_infer.value = 'Stop Infering'
            }


            if (infer_request) {
                infer_request = false;
                // createModel();
                startInference();
            }

        } else {
            // btn_infer.className = 'btn btn-danger btn-md';
            btn_infer.disabled = true;
            btn_infer.value = 'Model not valid or being reinitialized'
        }
    }

    setTimeout(function () {
        monitor();
    }, 100);
}


function start() {

    supported = detect_support();
    supported = true;
    load_weights();

    if (supported) {
        console.log('device & webgl supported');
        btn_infer.disabled = false;

        setTimeout(function () {

            run();

            monitor();

        }, 0);

    } else {
        console.log('device/webgl not supported')
        btn_infer.disabled = true;
    }



}
