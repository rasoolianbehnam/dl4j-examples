package org.deeplearning4j.examples.dataexamples.mytests;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.dataexamples.MyTestsPhoenix;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.IOException;

public class AutoencoderSimple {
    private static MultiLayerNetwork createNN() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.ADAGRAD)
            .activation(Activation.RELU)
            .l2(.0001)
            .updater(new Sgd(.05))
            .list()
            .layer(0, new DenseLayer.Builder().nIn(728).nOut(250).build())
            .layer(1, new DenseLayer.Builder().nIn(250).nOut(10).build())
            .layer(2, new DenseLayer.Builder().nIn(10).nOut(250).build())
            .layer(3, new OutputLayer.Builder().nIn(250).nOut(728)
                    .lossFunction(LossFunctions.LossFunction.MSE).build())
            .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        return net;
    }
    public static void main(String args[]) throws IOException {
        DataSetIterator dsi = new MnistDataSetIterator(100, 50000, false);

    }
}
