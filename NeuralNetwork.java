package basicnn;

import java.util.Vector;
import java.util.Arrays;

/*
 * @author tudorel
 */
public class NeuralNetwork {
    private String activatorFunction = "sigmoid";       // for NNcomprehensiveness
    private float learningRate = (float) 0.07;          // controlling how big the "learning jump" is
    private int iterations = 10000;                     // how many training runs; superceded by the "iter" parameter in the function below
    private int probingPoint = 1000;                    // every 1000th epoch log the data
    private int loggerIndex=0;
    private int percentageDone=0;
    private static int m=0;
    private float MSEAvgResult;
    private float MSEAvgSlope=1;
    private float[] individualSlopes;
    String [][] logger;                 // done this way since it's actually more computationally expensive to print 10 times per ieration
//    private FileWriter log;
//    private BufferedWriter bw;
    private int[] layerDistribution;                    // how the neurons are distributed across the layers; implicitly contains the number of layers also
    private Weights weights = new Weights(new int[]{2,3,1});             // automatically initialized to some random weight values; will be overwritten if user sends input
    Vector<FFCalcs> feedForwardCalcs = new Vector<FFCalcs>();


    public NeuralNetwork(float learningR, int iter, int[] neuronDistribution)
    {
        // hyperparameters
        learningRate = learningR;
        iterations = iter;
        percentageDone = iterations/10;
        layerDistribution = neuronDistribution;
        MSEAvgResult = 0;      // initialized to 1 since we want the initial passes to have maximal impact on the weights
        this.weights = new Weights(neuronDistribution);
    }
    
    
    public float[] forwardPropagation(float[] inputs)
    {
        FFCalcs buff;
        
        for(int i=0; i<layerDistribution.length-1; i++)
        {
            // we instantiate it here as to lose the link between the buff variable and the vector holding the values; otherwise we end up with a vector filled with only the latest value
            buff = new FFCalcs();
            
            // note that these two arrays are offset by 1 -> the sums of the first hidden layer are on index 0
            buff.sumsForLayer = new float[layerDistribution[i+1]];
            buff.resultsForLayer = new float[layerDistribution[i+1]];
            
            if(i!=0)    // for all the rest, we use the results of the previous layers
            {
                // we use "feedForwardCalcs.get(i-1).resultsForLayer" because when i=1, we create an entry on index 0; when we reach i=1, we need the result from index 0; for i=2, we need index 1; etc
                // (due to the offset previously mentioned)
                buff.sumsForLayer = weightsMultiply(weights.weightsEmergingFromLayer.get(i), feedForwardCalcs.get(i-1).resultsForLayer);
                buff.resultsForLayer = sigmoid(buff.sumsForLayer);
            }
            else        // for the first layer, we use the system inputs. Why you do that a TODOR!!!!!
            {
                buff.sumsForLayer = weightsMultiply(weights.weightsEmergingFromLayer.get(i), inputs);
                buff.resultsForLayer = sigmoid(buff.sumsForLayer);
            }
            
            //point 2
            
            feedForwardCalcs.add(buff);
        }
        
        return feedForwardCalcs.get(feedForwardCalcs.size()-1).resultsForLayer;       // return the final output of the network (given these inputs); which is the final element in the vector
    }
    
    
    
    public float[] backPropagation(float[] inputs, float[] outputs, int exampleCounter)
    {
        // calculate the output error:
        float[] outputError = new float[outputs.length];      // only for the final layer
        float[][] deltaOutputSum = new float[feedForwardCalcs.size()][];
        float[] calculatedOutputsForLayer, sumsForLayer;
        // 3-dimensional array:
        // 1st dimension: the number of layers
        // 2nd dimension: the number of neurons on each layer
        // 3rd dimension: the number of weights attached to each neuron
        float[][][] weightChanges = new float[feedForwardCalcs.size()][][];       // also known as the delta weights
        
        //for each layer in the network
        for(int k=feedForwardCalcs.size()-1; k >= 0; k--)
        {
            //buff.sumsForLayer = weightsMultiply(weights.weightsEmergingFromLayer.get(i), feedForwardCalcs.get(i-1).resultsForLayer);
            //buff.resultsForLayer = sigmoid(buff.sumsForLayer);
            calculatedOutputsForLayer = feedForwardCalcs.get(k).resultsForLayer;
            sumsForLayer = feedForwardCalcs.get(k).sumsForLayer;
            deltaOutputSum[k] = new float [calculatedOutputsForLayer.length];
            float[][] deltaWeights;
            
            // we use the results as the "fundament" for our delta weights since
            // at the end we're gonna multiply these values anyway
            // Logic: we use 'k-1' because we need the results from the previous layer
            if(k >= 1)       // if we haven't reached the first layer
                {// get the results from the previous layer
                deltaWeights = new float[feedForwardCalcs.get(k-1).resultsForLayer.length][];
                deltaWeights[0] = feedForwardCalcs.get(k-1).resultsForLayer;}     // temporarily used as a buffer for a later division
            else            // k == 0 => first layer
                {// take the actual inputs for this run
                deltaWeights = new float[inputs.length][];
                deltaWeights[0] = inputs;}     // temporarily used as a buffer for a later division

            
            if(k == feedForwardCalcs.size()-1)        // output error is only used in the last layer of the network
            {
                // calculate the output error
                for(int i=0; i<outputError.length; i++)     // for each neuron in the output
                {
                    outputError[i] = outputs[i] - calculatedOutputsForLayer[i];
                    deltaOutputSum[k][i] = oneToOneMultiply(sigmoidDerivative(sumsForLayer[i]), outputError[i])[0][0];
                }
                
                // point 1
            }
            else
            {
                float deltaOutputPrevLayer = 0;
                float[] weightsForNeuron = new float[calculatedOutputsForLayer.length];
                
                // we add up all the delta output sums from the previous layer, to get a total delta output sum
                for(int i=0; i<deltaOutputSum[k+1].length; i++)     // for each neuron in the layer
                    {deltaOutputPrevLayer += deltaOutputSum[k+1][i];}
                
                for(int i=0; i<calculatedOutputsForLayer.length; i++)     // for each neuron in the layer
                {
                    float[] tmp = weights.weightsEmergingFromLayer.get(k+1)[i];     // k - get the weights coming out of this layer; i - get the weights tied to this neuron
                    
                    // each neuron has several weights coming out of it; we add them all up to get the "weight of the neuron"
                    for(int j=0; j< tmp.length; j++)        
                        {weightsForNeuron[i] += tmp[j];}
                }
                
                // we do the first operation here: the division
                deltaOutputSum[k] = oneToOneDivide(new float[] {deltaOutputPrevLayer}, weightsForNeuron)[0];
                // we do the final part of the multiplication here
                deltaOutputSum[k] = normalMultiply(deltaOutputSum[k], sigmoidDerivative(sumsForLayer));
            }
            
            
            deltaWeights = oneToOneMultiply(deltaOutputSum[k], deltaWeights[0]);
            
            
            // we use the learning rate as an order of magnitude, to scale how drastic the changes in this iteration are
            for (int i=0; i<deltaWeights.length; i++)
            {
                deltaWeights[i] = oneToOneMultiply(deltaWeights[i], MSEAvgSlope)[0];       // previously was learningRate; 
            }         // learning rate not applied???
            
            weightChanges[k] = deltaWeights;
        }
        
        // updating the weights with new values
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for( int k=0; k <= feedForwardCalcs.size()-1; k++)         // for each layer of neurons
        {
            float[][] buff = weights.weightsEmergingFromLayer.get(k);   // get the neurons and all their weights
            
            for(int j=0; j<buff.length; j++)                // for each neuron
            {
                for(int l=0; l<buff[0].length; l++)         // for each weight emerging from  neuron j
                {
                    buff[j][l] += weightChanges[k][j][l];
                    
                    // *DON'T USE SINCE THE LEARNING RATE IS 0.07 SO IT WILL CLEAR ALL WEIGHT CHANGES!!!*
//                    buff[j][l] = (float) (Math.floor(buff[j][l] * 100.0) / 100.0);        // shave off all extra digits after the first two following the dot.
                }
            }

            weights.weightsEmergingFromLayer.set(k, buff);
            // printout the weight changes that happen in the first layer, for every passing iteration
//            if((k==0) && (m % 100 == 0))
//                {
//                 System.out.println("=============iteration "+m+"=============");
//                 System.out.println("buff display: "+Arrays.toString(buff[0]));
//                 System.out.println("buff display: "+Arrays.toString(buff[1]));
//                 System.out.println("=========================================");

                 // point 3
//                }
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        return outputError;
    }

    // take each element from one side it and multiply it with its counterpart from the other side
    public float[] normalMultiply(float[] a, float[] b)
    {
        float[] c = new float[a.length];
       
        for (int i = 0; i < a.length; i++)
        {
            c[i] = a[i] * b[i];
        }
       
        return c;
    }
     
    // take each column from the weights matrix and multiply it with the inputs matrix. Example:
    // for weights: w1 w2 w3
    //              w4 w5 w6
    // and inputs:  i1 i2
    // the multiplication will be:
    //      i1 * w1 + i2 * w4
    //      i1 * w2 + i2 * w5
    //      i1 * w3 + i2 * w6
    public float[] weightsMultiply(float[][] a, float[] b)
    {
        int columnsInA = a[0].length;
        int rowsInA = a.length;
        float[] c = new float[columnsInA];
       
        // we need to initialize our output array
        for (int i = 0; i < columnsInA; i++)
            {c[i] = 0;}
        
        for (int i = 0; i < columnsInA; i++)
        {
            for (int j = 0; j < rowsInA; j++)
            {
                // we are guaranteed that the number of rows in a matches the number of inputs coming in
                c[i] += a[j][i] * b[j];
            }
        }
        
        // normalizing results since we have an issue:
        // the sigmoidDerivative function can't process numbers smaller than -88,
        // so if the sumsForLayer is smaller than that value, the sDerivative()
        // returns NaN which essentially kills the whole run
        for (int i = 0; i < columnsInA; i++)
        {
            if(c[i] <-88)
            {
                // we tone it down a little since anyway -45 or -88 yield the same result: 0.0
                c[i] = -45;
            }
        }
       
        return c;
    }
    
    // multiply the array a with each element in array b. Example:
    // for a: [1,2,3]
    //     b: [1,1]
    // the result will be:
    //     c: [1,2,3,1,2,3]
    public float[][] oneToOneMultiply(float[] a, float... b)
    {
        float[][] c = new float[b.length][a.length];
        
        for(int i=0; i < b.length; i++)
        {
            for(int j=0; j < a.length; j++)
            {
                c[i][j] = a[j]*b[i];
            }
        }
        
        return c;
    }
    
    
    public float[][] oneToOneDivide(float[] a, float... b)
    {
        float[][] c = new float[b.length][a.length];
        
        for(int i=0; i < b.length; i++)
        {
            for(int j=0; j < a.length; j++)
            {
                c[i][j] = a[j]/b[i];
            }
        }
        
        // if we do a division of type:
        // x / [a, b, c ]
        // (where we divide one number by a whole array)
        // we need to transpose the result, otherwise the result will be:
        // c[x][1]:
        // [a]
        // [b]
        // ...
        // [x] -> xth column
        if(c[0].length == 1)
        {c=transposeMatrix(c);}
        
        return c;
    }
    
     
    // implements: def sigmoid(z):
    //              return 1.0/(1.0+np.exp(-z))
    public float[] sigmoid(float... z)
    {
        float[] result = new float [z.length];
        
        for(int i=0; i<z.length; i++)
        {
         result[i] = (float) (1.0/(1.0 + Math.exp((double)-z[i])));
//         result[i] = (float) (Math.floor(result[i] * 100.0) / 100.0);        // shave off all extra digits after the first two following the dot.
        }
        
        return result;
    }
    
    public float[] sigmoidDerivative(float... z)
    {
        // point 4
        float[] result = new float[z.length];
        
        // equation: (e^-x)/(1+e^-x)^2
        // we use this method as opposed to "sigmoid(x) / (1-sigmoid(x))" because
        // that method deviates from the value that we expect for the sigmoid derivative
        // !!! this methods has a tendency to get super close to 0, which sometimes causes a NaN
        // value to appear, so we need to do a roundoff !!!
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for(int i=0; i<z.length; i++)
        {
            // (e^-x)
            float var1 = (float) Math.exp(-z[i]);
            // (1+e^-x)^2
            float var2 = (float) Math.pow((1+var1), 2);
            result[i] =  var1 / var2;
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
        for(int i=0; i< result.length; i++)
        {
            if(result[i] != result[i])
            {
                System.out.println("It was the sigmoid derivative!");
            }
        }
        
        return result;
        
    }
    
    public static float[][] transposeMatrix(float [][] m){
        float[][] temp = new float[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }
    
    
    // breakdown of the input variable:
    // examples[0][] = {... , ... , ...}        // array of inputs on row 1
    // examples[1][] = {... , ... , ...}        // array of exspected outputs on row 2
    public void learn(float [][][] examples)
    {
        boolean endLearning = false;
        logger = new String [probingPoint*10][examples.length+4];
//        float previousMSEResult=-1;      // used for accelerated learning
        
        // point 5
                    
        for (m = 0; m < iterations; m++)        // # we declare the index globally so that we can see at any point at which iteration we are
        //  we use the 1 value to get over the fencepost problem
//        while((MSEAvgSlope > 0.7) || (MSEAvgSlope == 1))       // the perfect run had a slope of 0.37703308; a successful run (just 1 result is off had a slope of 0.56912845
        {
            float[][] errors = new float[examples.length][];
            float[][] results = new float[examples.length][];
            float[] individualMSE = new float[examples.length];
            individualSlopes = new float[examples.length];
            
            if(endLearning)     // we reached a minima -> we're done
                break;
            
            for (int i=0; i < examples.length; i++)                // (float[][] example : examples)
            {
                results[i] = forwardPropagation(examples[i][0]);
                errors[i] = backPropagation(examples[i][0], examples[i][1], i);
                
                // we wanna do at least one iteration with the mean error from the previous
                // run; hence this check is positioned here
//                if(MSEResult[i] <= 0.0010)
//                    {endLearning=true;
//                     break;}
                
                
                // Equation for the quadratic cost function - C - (also known as the mean squared error: MSE)
                // C = 1/2*n * Sum ||(expected out - actual out)||^2
                //
                // n = total number of training inputs (how many examples we give on each iteration);
                // (Sum (||...||))^2 = the difference between the expected result and the actual result, 
                // calculated the  vector length for that result, then Summed up all the results, then squared;
                // ||X|| = if X is a vector [1, 2], then the result would be: SQRT(1^2 + 2^2)
                //
                // We also calculate the slope of the error, based on the formula:
                // 
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                // ||X||
                float tmp = vectorLength(errors[i]);
                // (Sum (||X||))^2
                individualMSE[i] = (float) Math.pow(tmp, 2);
                individualSlopes[i] = 2*tmp;
                MSEAvgSlope += individualSlopes[i];

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                

                // reset the previous calculations
                // (we don't need to reset other variables since their scope is function-wide)
                feedForwardCalcs = new Vector<>();
            }
            
            for(int i=0; i<individualMSE.length; i++)
                {MSEAvgResult += individualMSE[i];}
            
            // (1/ 2*n) * ...
            MSEAvgResult /= examples.length;
            MSEAvgSlope /= examples.length;
            
            if(m%probingPoint == 0)
            {
                for(int i=0; i<examples.length; i++)
                {
                    if(examples[i][0][0] * examples[i][0][1]< 10)
                        {logger[loggerIndex][i]="Run: "+m+"; input: "+(int)examples[i][0][0]+" & "+(int)examples[i][0][1]+"; calc. output: "+determineMaxRes(results[i])+" errors: "+Arrays.toString(results[i])+'\n';}      // +" MSE: "+MSEResult
                    else
                        {try{
                            logger[loggerIndex][i]="Run: "+m+"; input: "+(int)examples[i][0][0]+" & "+(int)examples[i][0][1]+"; calc. output: "+Arrays.toString(determineMaxResTwoDigits(results[i]))+" errors: "+Arrays.toString(results[i])+'\n';
                        }
                        catch(Exception e)
                        {
                            e.printStackTrace();
                        }
                        }      // +" MSE: "+MSEResult}
                }
            
                logger[loggerIndex][examples.length]=" Avg. MSE: "+MSEAvgResult+'\n';
                logger[loggerIndex][examples.length+1]=" Avg. Slope: "+MSEAvgSlope+'\n';
                logger[loggerIndex][examples.length+2]=" Individual slopes: "+Arrays.toString(individualSlopes)+'\n';
                logger[loggerIndex][examples.length+3]="==================================="+'\n';

                loggerIndex++;
            }
            
//            if(m%50000 ==0)
//            {
//                System.out.println(Arrays.toString(logger[loggerIndex-1]));
//            }
            
            if((m%percentageDone ==0) && (m!=0))
            {
                int progressCalculator = m/percentageDone*10;  // this will give us the actual percent: 10%, 20%, etc. 
                System.out.println(progressCalculator+"% done.");
            }

//            m++;        // just when using the "while" loop
        }
        
        // wait for the learning to finish and then print it out
         for (int i = 0; i < loggerIndex; i++)        // # we declare the index globally so that we can see at any point at which iteration we are
        {
            System.out.println(Arrays.toString(logger[i]));
        }
        
        // point 6
    }
    
    public float[] run(float[] input)
    {
        float[] res = forwardPropagation(input);
        feedForwardCalcs = new Vector<>();      // reset the calculations for the next round
        
        return res;
    }

    
    private void printMatrix(float[][] matrix)
    {
        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[i].length; j++)
            {
                System.out.print(matrix[i][j] + "  ");
            }
            
            System.out.println();
        }
    }

//    private void writeToFile(float value, int iteration)
//    {
//        try
//            {
//             log = new FileWriter("/home/tbalus/Desktop/jlog",true);
//             log.write(iteration+", "+value+"\n");
//            }
//        catch (IOException ex)
//            {ex.printStackTrace();}
//        finally
//        {
//            
//            try
//            {
//            if (bw != null)
//                bw.close();
//
//            if (log != null)
//                log.close();
//            }
//        catch (IOException ex)
//            {ex.printStackTrace();}
//        }   
//    }
    
     public static int determineMaxRes(float[] result) {
        float res=-1;
        int pos=-1;
        
        for(int i=0; i<result.length; i++)
        {
            if(result[i] > res)
            {
                res = result[i];
                pos = i;
            }
        }
        
        return pos;
    }
     
     public static int[] determineMaxResTwoDigits(float[] result) {
        float res=-1;
        int[] pos = new int[]{-1, -1};
        
        for(int i=0; i<result.length; i++)
        {
            if(result[i] > res)
            {
                res = result[i];
                pos[1] = pos[0];
                pos[0] = i;
            }
        }
        
        // arrange them if they're not in order
        if(pos[0]>pos[1])
        {
            int tmp = pos[0];
            pos[0] = pos[1];
            pos[1] = tmp;
        }
        
        return pos;
    }
     
     // implements the vector length function:
     // ||X|| = if X is a vector [1, 2], then the result would be: SQRT(1^2 + 2^2)
     public float vectorLength(float[] outputErrorDiff)
     {
        float res = 0;

        // (X1^2 + X2^2)
        for (float error: outputErrorDiff)
        {
            res += Math.pow(error, 2);
        }

        res = (float) Math.sqrt(res);
         
        return res;
     }
}




// POINT 1:
// testing to see if the NN actually learns or not
//                if((inputs[1] == 1))        //  && (m%100 == 0)
//                {
//                    System.out.println("+++++++++++++++++++++++++++++++++");
//                    System.out.println(m+" For inputs: 1 and 1, we have the calculated output: "+calculatedOutputsForLayer[1]);
//                    System.out.println(m+" For inputs: 1 and 1, we have the weights: "+Arrays.toString(weights.weightsEmergingFromLayer.get(0)[0]));
//                    System.out.println(m+" For inputs: 1 and 1, we have the weights: "+Arrays.toString(weights.weightsEmergingFromLayer.get(0)[1]));
//                    System.out.println("+++++++++++++++++++++++++++++++++");
//                }
// +++++++++++++++++++++++++++++++++++++++++


// POINT 2:
// round off extra digits:
//            for(int j=0; j<layerDistribution[i+1]; j++)
//            {
//                buff.sumsForLayer[j] = (float) (Math.floor(buff.sumsForLayer[j] * 100.0) / 100.0);
//                buff.resultsForLayer[j] = (float) (Math.floor(buff.resultsForLayer[j] * 100.0) / 100.0);
//            }
// +++++++++++++++++++++++++++++++++++++++++


// POINT 3:
//                 System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
//                 printMatrix(weights.weightsEmergingFromLayer.get(weights.weightsEmergingFromLayer.size()-2));
//                 System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
//                 System.out.println();
// +++++++++++++++++++++++++++++++++++++++++


// POINT 4:
// ************************************
        // this is just another way of doing the sigmoid derivative. Formula:
        // sigmoid (x) * (1 - sigmoid(x))
        //
        // we don't use it since it would take more computational power (a function call)
        // than our other implementation
        // ************************************
//        float[] sigmoid = sigmoid(z);
        
//        for(int i=0; i<sigmoid.length; i++)
//        {
//            float tmp = sigmoid[i];
//            result[i] = tmp * ((float)1 - tmp);
//        }
// +++++++++++++++++++++++++++++++++++++++++


// POINT 5:
//        System.out.println("Initial weights:");
//        for(int i=0; i<weights.weightsEmergingFromLayer.size(); i++)
//        {
//            printMatrix(weights.weightsEmergingFromLayer.get(i));
//            System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
//        }
//        System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
//        System.out.println();
//        System.out.println();
//        System.out.println();
// +++++++++++++++++++++++++++++++++++++++++


// POINT 6:
//        System.out.println("Final weights:");
//        for(int i=0; i<weights.weightsEmergingFromLayer.size(); i++)
//        {
//            printMatrix(weights.weightsEmergingFromLayer.get(i));
//            System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
//        }
//        System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
// +++++++++++++++++++++++++++++++++++++++++
