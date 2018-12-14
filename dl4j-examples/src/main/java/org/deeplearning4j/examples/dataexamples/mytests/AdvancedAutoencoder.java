package org.deeplearning4j.examples.dataexamples.mytests;

import org.apache.commons.io.FileUtils;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.net.URL;

public class AdvancedAutoencoder {
    public static File downloadFile(File cache, String url) throws Exception {
        String splitPath[] = url.split("/");
        String fileName = splitPath[splitPath.length-1];
        System.out.println(fileName);
        File tmpZip = new File(cache, fileName);
        tmpZip.delete();
        System.out.println("Downloading file...");
        FileUtils.copyURLToFile(new URL(url), tmpZip);
        System.out.println("Finished Downloading.");
        return tmpZip;
    }

    public static void main(String args[]) {
        File dataFile = new File(cache, "/aisdk_20171001.csv");
        if (!dataFile.exists()) {
            String remote = "http://blob.deeplearning4j.org/datasets/aisdk_20171001.csv.zip";
            File tmpZip = downloadFile(cache, remote);
            System.out.println("Decompressing file...");
            ArchiveUtils.unzipFileTo(tmpZip.getAbsolutePath(), cache.getAbsolutePath());
            tmpZip.delete();
            System.out.println("Done.");
        } else {
            System.out.println("File already Exists.");
        }
    }
}
