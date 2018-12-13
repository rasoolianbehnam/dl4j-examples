package org.deeplearning4j.examples.dataexamples;

import nu.pattern.OpenCV;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.util.ImageUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import javafx.scene.image.*;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.HighGui;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class MyTests {
    public static Random rnd = new Random(123);
    private  static void writeNDArrayToImage(INDArray array, BufferedImage image) {
        for (int i = 0; i < array.shape()[2]; i++) {
            for (int j = 0; j < array.shape()[3]; j++) {
                int g, r, b;
                g = (int) array.getDouble(0, 0, i, j);
                r = (int) array.getDouble(0, 1, i, j);
                b = (int) array.getDouble(0, 2, i, j);
                Color color = new Color(r, g, b);
                image.setRGB(i, j, color.getRGB());
            }
        }
    }
    private static void drawImage(INDArray generated, BufferedImage drawing) {
        /*Assume generated is of size N * 3 where N = w*h*/
        int N = (int) generated.shape()[0];
        int w = drawing.getWidth();
        int h = drawing.getHeight();
        if (N != w * h) throw new IllegalArgumentException("generated and image cannot be put together");
        generated = generated.mul(generated.lte(1)).add(generated.gt(1).mul(1));
        generated.muli(generated.gt(0));
        //generated.putWhereWithMask(generated.lt(0), 0);
        //generated.putWhereWithMask(generated.gt(1), 1);
        writeNDArrayToImage(generated.mul(255).reshape(1, w, h, 3).swapAxes(3, 2).swapAxes(2, 1), drawing);
    }
    public static void main(String args[]) throws IOException, InterruptedException {
        int W = 400;
        int H = 400;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        BufferedImage drawing = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        NativeImageLoader loader = new NativeImageLoader(W, H, 3);
        INDArray originalImage =
                loader.asMatrix(new File("/Users/behnamrasoolian/git/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/Mona_Lisa.png"));
        INDArray anotherImage =
                loader.asMatrix(new File("/Users/behnamrasoolian/Dropbox/Photos/IMG_20140220_034619288.jpg"));
        writeNDArrayToImage(originalImage, img);
        writeNDArrayToImage(anotherImage, drawing);

        JFrame frame = new JFrame();
        ImageIcon icon = new ImageIcon(img);
        ImageIcon icon2 = new ImageIcon(drawing);
        frame.setLayout(new GridLayout(1, 2));
        frame.setSize(W*2, H);
        JLabel lbl = new JLabel();
        JLabel lbl2 = new JLabel();
        lbl.setIcon(icon);
        lbl2.setIcon(icon2);

        frame.add(lbl, 0, 0);
        frame.add(lbl2,1, 1);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        INDArray xyOut = Nd4j.zeros(W*H, 2);
        for (int i = 0; i < W; i++) {
            double xp = scaleXY(i,W);
            for (int j = 0; j < H; j++) {
                int index = i + W * j;
                double yp = scaleXY(j,H);

                xyOut.put(index, 0, xp); //2 inputs. x and y.
                xyOut.put(index, 1, yp);
            }
        }

        MultiLayerNetwork net = createNN();
        net.setListeners(new ScoreIterationListener(10));

        int batchSize = 1000;
        int numBatches = 5;

        for (int t=0; t<1000; t++) {
            for (int i=0; i<numBatches; i++) {
                DataSet ds = generateDataSet(originalImage, batchSize);
                net.fit(ds);
            }
            INDArray predicted = net.output(xyOut);
            drawImage(predicted, drawing);
            lbl2.setIcon(new ImageIcon(drawing));
        }

    }
    private static DataSet generateDataSet(INDArray image, int batchSize) {
        /*Assuming image is n*c*w*h */
        int w = (int) image.shape()[2];
        int h = (int) image.shape()[3];

        INDArray xy = Nd4j.zeros(batchSize, 2);
        INDArray out = Nd4j.zeros(batchSize, 3);

        for (int index = 0; index < batchSize; index++) {
            int i = rnd.nextInt(w);
            int j = rnd.nextInt(h);
            double xp = scaleXY(i,w);
            double yp = scaleXY(j,h);

            xy.put(index, 0, xp); //2 inputs. x and y.
            xy.put(index, 1, yp);
            //System.out.println("output shape is " + Arrays.toString(out.shape()));
            out.put(index, 0, image.getDouble(0, 0, i, j)/255.);  //3 outputs. the RGB values.
            out.put(index, 1, image.getDouble(0, 1, i, j)/255.);
            out.put(index, 2, image.getDouble(0, 2, i, j)/255.);
        }
        return new DataSet(xy, out);
    }

    /**
     * Build the Neural network.
     */
    private static MultiLayerNetwork createNN() {
        int seed = 2345;
        double learningRate = 0.05;
        int numInputs = 2;   // x and y.
        int numHiddenNodes = 100;
        int numOutputs = 3 ; //R, G and B value.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    /**
     * Make sure the color values are >=0 and <=1
     */
    private static double capNNOutput(double x) {
        double tmp = (x<0.0) ? 0.0 : x;
        return (tmp > 1.0) ? 1.0 : tmp;
    }

    /**
     * scale x,y points
     */
    private static double scaleXY(int i, int maxI){
        return (double) i / (double) (maxI - 1) -0.5;
    }
}
