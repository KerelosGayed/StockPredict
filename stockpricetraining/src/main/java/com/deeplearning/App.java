package com.deeplearning;

// Model & readers
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import java.net.*;
//The normalizers
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize; // Kind of obsolete for this test (Ruined MSE)

//The learning algorithms (All brought good results when fine-tuned, usually a MSE of ~2.3 when trained at 100k iterations, 1.2M iterations will half these results)
import org.nd4j.linalg.learning.config.Sgd; // Fine-tuned best MSE: 0.00001 LR = 2.51398876362572
import org.nd4j.linalg.learning.config.Adam; // Fine-tuned best MSE: (0.001,0.9,0.999, 0.000000001) (LR, B1, B2, Epsilon) = 2.314390886108584
import org.nd4j.linalg.learning.config.RmsProp; // Fine-tuned best MSE: (0.001, 0.9, 1e-8) (LR, Decay rate, Epsilon) = 2.4299205385932696
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet;

//Extra for file reading, exceptions, data sorting
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class App {

    public static void main(String[] args) throws IOException, InterruptedException {
        int numInputs = 4; // Number of input features
        int numOutputs = 1; // Number of output features
        int numHiddenNodes = 10; // Number of hidden nodes

        // Define the schema
        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("Date")
                .addColumnDouble("Open")
                .addColumnDouble("High")
                .addColumnDouble("Low")
                .addColumnDouble("Close")
                .addColumnInteger("Volume")
                .build();

        // Define the transformation process
        TransformProcess transformProcess = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Date") // Remove Date column because normalizer hates dashes in the dates
                .build();

        // Load and preprocess the data
        CSVRecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(new File("C:\\Users\\ZodoBoot\\OneDrive - Limestone DSB\\ICS3U\\SPY.csv")));
        // Please enter directory to the attached "SPY.csv" or any Yahoo Finance
        // Historical
        // Data sheets, you must remove "Adj Close" column manually.

        // Convert to a list of lists of writables
        List<List<Writable>> originalData = new ArrayList<>();
        while (recordReader.hasNext()) {
            originalData.add(recordReader.next());
        }

        // Debug: Print original data
        System.out.println("Original Data:");
        for (List<Writable> record : originalData) {
            System.out.println(record);
        }

        // Transform the data
        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, transformProcess);

        // Debug: Print processed data
        System.out.println("Processed Data:");
        for (List<Writable> record : processedData) {
            System.out.println(record);
        }

        // Ensure processed data is not empty
        if (processedData.isEmpty()) {
            throw new RuntimeException("Processed data is empty");
        } else {
            System.out.println("Processed data is ready for further steps");
        }

        // Create a RecordReader from the processed data
        RecordReader transformedRecordReader = new CollectionRecordReader(processedData);

        // Create DataSetIterator
        int batchSize = 32;
        int labelIndex = 3; // 'Close' column index after removing 'Date'

        DataSetIterator trainData = new RecordReaderDataSetIterator(transformedRecordReader, batchSize, labelIndex,
                labelIndex, true);

        // Normalize data
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        if (trainData.hasNext()) {
            System.out.println("Training data has records, proceeding to fit normalizer");
            normalizer.fit(trainData);
            trainData.setPreProcessor(normalizer);
        } else {
            throw new RuntimeException("No data found in trainData");
        }
        System.out.println("Testing");
        // Reset iterator after fitting normalizer
        trainData.reset();

        // Debug: Iterate through the training data to ensure it's loaded correctly
        System.out.println("Training Data:");
        while (trainData.hasNext()) {
            DataSet dataSet = trainData.next();
            System.out.println(dataSet);
        }
        // Define the network configuration
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(123) // Random seed for reproducibility
                .weightInit(WeightInit.XAVIER) // Weight initialization
                .updater(new Adam(0.001, 0.9, 0.999, 0.000000001)) // Optimization algorithm (Results: 0.0001 best for
                                                                   // Sgd, 0.001 for Adam)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build());

        model.init();
        // Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        // Configure where the network information (gradients, score vs. time etc) is to
        // be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File), for saving
                                                                // and loading later

        // Attach the StatsStorage instance to the UI: this allows the contents of the
        // StatsStorage to be visualized
        uiServer.attach(statsStorage);

        // Then add the StatsListener to collect this information from the network, as
        // it trains
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(70000)); // Access UI Data here:
                                                                                               // http://localhost:9000/train/overview

        // Train the model
        for (int epoch = 0; epoch < 7000; epoch++) {
            model.fit(trainData);
        }
        // Reset the iterator
        trainData.reset();

        double sumSquaredErrors = 0.0;
        int totalSamples = 0;

        // Evaluate the model
        while (trainData.hasNext()) {
            DataSet next = trainData.next();
            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();

            INDArray predicted = model.output(features, false);

            // Compute squared errors
            INDArray diff = labels.sub(predicted);
            INDArray squaredDiff = diff.mul(diff);

            sumSquaredErrors += squaredDiff.sumNumber().doubleValue();
            totalSamples += labels.size(0);
        }

        // Calculate Mean Squared Error
        double mse = sumSquaredErrors / totalSamples;

        // Output evaluation statistics
        System.out.println("Mean Squared Error (MSE): " + mse);
        // Save the trained model
        File locationToSave = new File("StockPricePredictorModel.zip");
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        System.out.println("Model training complete and saved to " + locationToSave.getPath());

        Thread.sleep(6000);

        // Evaluate the model
        trainData.reset();
        RegressionEvaluation eval = new RegressionEvaluation();
        while (trainData.hasNext()) {
            DataSet batch = trainData.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            INDArray predicted = model.output(features, true);
            eval.eval(labels, predicted);
        }

        System.out.println(eval.stats());
        Thread.sleep(5000);

        // Visualize predictions vs actual values
        trainData.reset();
        List<Double> residuals = new ArrayList<>();
        while (trainData.hasNext()) {
            DataSet batch = trainData.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            INDArray predicted = model.output(features, true);

            System.out.println("Actual: " + labels);
            System.out.println("Predicted: " + predicted);
            eval.eval(labels, predicted);

            // Compute residuals
            INDArray residual = labels.sub(predicted);
            for (int i = 0; i < residual.length(); i++) {
                residuals.add(residual.getDouble(i));
            }
        }
        double sumSquaredErrorsEval = 0.0;
        for (double residual : residuals) {
            sumSquaredErrorsEval += Math.pow(residual, 2);
        }
        double standardError = Math.sqrt(sumSquaredErrorsEval / residuals.size());

        // Calculate the margin of error (95% confidence interval)
        double marginOfError = 1.96 * standardError;
        String firstMargin = eval.stats();

        System.out.println("Calculated margin of Error: " + marginOfError);
        System.out.println(eval.stats());
        System.out.println("Beginning another evaluation test...");
        sumSquaredErrorsEval = 0.0;
        residuals.clear();
        standardError = 0.0;
        marginOfError = 0.0;
        Thread.sleep(5000);

        // Visualize predictions vs actual values (Second time)
        trainData.reset();
        while (trainData.hasNext()) {
            DataSet batch = trainData.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            INDArray predicted = model.output(features, false);

            System.out.println("Actual: " + labels);
            System.out.println("Predicted: " + predicted);
            eval.eval(labels, predicted);
            // Compute residuals (Second time)
            INDArray residual = labels.sub(predicted);
            for (int i = 0; i < residual.length(); i++) {
                residuals.add(residual.getDouble(i));
            }
        }
        sumSquaredErrorsEval = 0.0;
        for (double residual : residuals) {
            sumSquaredErrorsEval += Math.pow(residual, 2);
        }
        standardError = Math.sqrt(sumSquaredErrorsEval / residuals.size());

        // Calculate the margin of error (95% confidence interval)
        marginOfError = 1.96 * standardError;

        System.out.println("Second margin of error (Testing mode & given 95% confidence interval): ");
        System.out.println(eval.stats());
        System.out.println("For comparison, first margin of error:\n"
                + firstMargin);
        Thread.sleep(6000);
    }
}
