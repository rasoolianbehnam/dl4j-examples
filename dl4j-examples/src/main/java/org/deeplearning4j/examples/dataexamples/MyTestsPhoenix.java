package org.deeplearning4j.examples.dataexamples;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MyTestsPhoenix {
    public static void showImage(BufferedImage img) {
        int W = img.getWidth();
        int H = img.getHeight();
        JFrame frame = new JFrame();
        ImageIcon icon = new ImageIcon(img);
        frame.setLayout(new GridLayout(1, 1));
        frame.setSize(W*2, H);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl, 0, 0);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
    public static void showImage(INDArray arr) {
        /*Assume image is of size W*H*c */
        int W = (int) arr.shape()[0];
        int H = (int) arr.shape()[1];
        BufferedImage img = new BufferedImage(H, W, BufferedImage.TYPE_INT_RGB);
        writeNDArrayToImage(arr, img);
        showImage(img);
    }
    public  static void writeNDArrayToImage(INDArray array, BufferedImage image) {
        for (int i=0; i<array.shape()[2]; i++) {
            for (int j=0; j<array.shape()[3]; j++) {
                int g, r, b;
                g = (int) array.getDouble(0, 0, i, j);
                r = (int) array.getDouble(0, 1, i, j);
                b = (int) array.getDouble(0, 2, i, j);
                Color color = new Color(r, g, b);
                image.setRGB(j, i, color.getRGB());
            }
        }
    }
    public static void main(String args[]) throws IOException {
        int W = 400;
        int H = 400;
        BufferedImage img = new BufferedImage(H, W, BufferedImage.TYPE_INT_RGB);
        BufferedImage drawing = new BufferedImage(H, W, BufferedImage.TYPE_INT_RGB);
        NativeImageLoader loader = new NativeImageLoader(W, H, 3);
        INDArray originalImage =
            loader.asMatrix(new File("/home/bzr0014/git/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/Mona_Lisa.png"));
        INDArray anotherImage =
            loader.asMatrix(new File("/home/bzr0014/Pictures/Mohammad Reza/20160905_120320.jpg"));
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

    }
}
