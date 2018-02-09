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

// import {
//     InputProvider
// } from './data/input_provider';
// import {
//     Tensor
// } from './graph/graph';
// import {
//     Optimizer
// } from './graph/optimizers/optimizer';
// import {
//     CostReduction,
//     FeedEntry,
//     Session
// } from './graph/session';
// import {
//     NDArrayMath
// } from './math/math';
// import {
//     NDArray,
//     Scalar
// } from './math/ndarray';

var dl = deeplearn;
var CostReduction = dl.CostReduction;

const DEFAULT_EVAL_INTERVAL_MS = 1500;
const DEFAULT_COST_INTERVAL_MS = 500;
const DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// export interface GraphRunnerEventObserver {
//     batchesTrainedCallback ? : (totalBatchesTrained: number) => void;
//     avgCostCallback ? : (avgCost: Scalar) => void;
//     metricCallback ? : (metric: NDArray) => void;
//     inferenceExamplesCallback ? :
//         (feeds: FeedEntry[][], inferenceValues: NDArray[]) => void;
//     inferenceExamplesPerSecCallback ? : (examplesPerSec: number) => void;
//     trainExamplesPerSecCallback ? : (examplesPerSec: number) => void;
//     totalTimeCallback ? : (totalTimeSec: number) => void;
//     doneTrainingCallback ? : () => void;
// }

var MetricReduction = {
    SUM: 0,
    MEAN: 1
}

/**
 * A class that drives the training of a graph model given a dataset. It allows
 * the user to provide a set of callbacks for measurements like cost, accuracy,
 * and speed of training.
 */
class GraphRunner {
    // private costTensor: Tensor;
    // private trainFeedEntries: FeedEntry[];
    // private batchSize: number;
    // private optimizer: Optimizer;
    // private currentTrainLoopNumBatches: number | undefined;
    // private costIntervalMs: number;

    // private metricTensor: Tensor | undefined;
    // private metricFeedEntries: FeedEntry[] | undefined;
    // private metricBatchSize: number | undefined;
    // private metricReduction: MetricReduction;
    // private metricIntervalMs: number;

    // private inferenceTensor: Tensor;
    // private inferenceFeedEntries: FeedEntry[] | undefined;
    // private inferenceExampleIntervalMs: number;
    // private inferenceExampleCount: number;

    // // Runtime information.
    // private isTraining: boolean;
    // private totalBatchesTrained: number;
    // private batchesTrainedThisRun: number;
    // private lastComputedMetric: NDArray;

    // private isInferring: boolean;
    // private lastInferTimeoutID: number;
    // private currentInferenceLoopNumPasses: number | undefined;
    // private inferencePassesThisRun: number;

    // private trainStartTimestamp: number;
    // private lastCostTimestamp = 0;
    // private lastEvalTimestamp = 0;

    // private lastStopTimestamp: number | null;
    // private totalIdleTimeMs = 0;

    // private zeroScalar: Scalar;
    // private metricBatchSizeScalar: Scalar;

    constructor(
        math, session,
        eventObserver) {
        this.math = math;
        this.session = session;
        this.eventObserver = eventObserver;
        this.lastCostTimestamp = 0;
        this.lastEvalTimestamp = 0;
        this.totalIdleTimeMs = 0;

        this.resetStatistics();
        this.zeroScalar = Scalar.new(0);
    }

    resetStatistics() {
        this.totalBatchesTrained = 0;
        this.totalIdleTimeMs = 0;
        this.lastStopTimestamp = null;
    }

    /**
     * Start the training loop with an optional number of batches to train for.
     * Optionally takes a metric tensor and feed entries to compute periodically.
     * This can be used for computing accuracy, or a similar metric.
     */
    infer(
        inferenceTensor, inferenceFeedEntries,
        inferenceExampleIntervalMs = DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS,
        inferenceExampleCount = 5, numPasses = null) {
        if (this.eventObserver.inferenceExamplesCallback == null &&
            this.eventObserver.inferenceExamplesPerSecCallback == null) {
            throw new Error(
                'Cannot start inference loop, no inference example or ' +
                'examples/sec observer provided.');
        }

        // Make sure the feed values are providers, and not NDArrays.
        for (let i = 0; i < inferenceFeedEntries.length; i++) {
            const feedEntry = inferenceFeedEntries[i];

            if (feedEntry.data instanceof NDArray) {
                throw new Error(
                    'Cannot start inference on the model runner with feed entries of ' +
                    'type NDArray. Please use InputProviders.');
            }
        }

        this.inferenceExampleIntervalMs = inferenceExampleIntervalMs;
        this.inferenceTensor = inferenceTensor;
        this.inferenceFeedEntries = inferenceFeedEntries;
        this.inferenceExampleCount = inferenceExampleCount;
        this.currentInferenceLoopNumPasses = numPasses;
        if (!this.isInferring) {
            this.inferencePassesThisRun = 0;
            requestAnimationFrame(() => this.inferNetwork());
        }
        this.isInferring = true;
    }

    inferNetwork() {
        if (!this.isInferring ||
            this.inferencePassesThisRun === this.currentInferenceLoopNumPasses) {
            return;
        }

        this.math.scope((keep, track) => {
            const feeds = [];
            const inferenceValues = [];

            const start = performance.now();
            for (let i = 0; i < this.inferenceExampleCount; i++) {
                // Populate a new FeedEntry[] populated with NDArrays.
                const ndarrayFeedEntries = [];
                for (let j = 0; j < this.inferenceFeedEntries.length; j++) {
                    const feedEntry = this.inferenceFeedEntries[j];
                    const nextCopy =
                        (feedEntry.data).getNextCopy(this.math);

                    ndarrayFeedEntries.push({
                        tensor: feedEntry.tensor,
                        data: track(nextCopy)
                    });
                }
                feeds.push(ndarrayFeedEntries);
                inferenceValues.push(
                    this.session.eval(this.inferenceTensor, ndarrayFeedEntries));
            }

            if (this.eventObserver.inferenceExamplesPerSecCallback != null) {
                // Force a GPU download, since inference results are generally needed on
                // the CPU and it's more fair to include blocking on the GPU to complete
                // its work for the inference measurement.
                inferenceValues[inferenceValues.length - 1].getValues();

                const inferenceExamplesPerSecTime = performance.now() - start;

                const examplesPerSec =
                    (this.inferenceExampleCount * 1000 / inferenceExamplesPerSecTime);
                this.eventObserver.inferenceExamplesPerSecCallback(examplesPerSec);
            }

            if (this.eventObserver.inferenceExamplesCallback != null) {
                this.eventObserver.inferenceExamplesCallback(feeds, inferenceValues);
            }
            this.inferencePassesThisRun++;

        });
        this.lastInferTimeoutID = window.setTimeout(
            () => this.inferNetwork(), this.inferenceExampleIntervalMs);
    }

    stopInferring() {
        this.isInferring = false;
        window.clearTimeout(this.lastInferTimeoutID);
    }

    isInferenceRunning() {
        return this.isInferring;
    }

    computeMetric() {
        if (this.metricFeedEntries == null) {
            throw new Error('Cannot compute metric, no metric FeedEntries provided.');
        }

        let metric = this.zeroScalar;

        return this.math.scope((keep) => {
            for (let i = 0; i < this.metricBatchSize; i++) {
                const metricValue =
                    this.session.eval(this.metricTensor, this.metricFeedEntries);

                metric = this.math.add(metric, metricValue);
            }

            if (this.metricReduction === MetricReduction.MEAN) {
                metric = this.math.divide(metric, this.metricBatchSizeScalar);
            }

            return metric;
        });
    }

    getTotalBatchesTrained() {
        return this.totalBatchesTrained;
    }

    getLastComputedMetric() {
        return this.lastComputedMetric;
    }

    setMath(math) {
        this.math = math;
    }

    setSession(session) {
        this.session = session;
    }

    setInferenceTensor(inferenceTensor) {
        this.inferenceTensor = inferenceTensor;
    }

    setInferenceExampleCount(inferenceExampleCount) {
        this.inferenceExampleCount = inferenceExampleCount;
    }
}
