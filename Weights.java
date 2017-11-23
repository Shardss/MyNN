package basicnn;

import java.util.Arrays;
import java.util.Random;
import java.util.Vector;

/*
 * @author tudorel
 */
class Weights {
    Vector<float[][]> weightsEmergingFromLayer = new Vector<float[][]>();
    
    // an array containing the numbers of neurons for each layer
    // (implicitly the number of layers is denoted by the size of the array)
    public Weights(int[] neurons)
    {
        Random r = new Random();
        float[][] buff;
        
        
        for (int i=0; i<neurons.length-1; i++)          // for each layer
        {
            buff = new float[neurons[i]][neurons[i+1]];
            for (int j=0; j<neurons[i]; j++)            // we take each neuron of the current layer
            {
                for (int k=0; k<neurons[i+1]; k++)      // and pair it with each neuron of the next layer
                {
                    // generating random weights
                    buff[j][k] = r.nextFloat();
                    buff[j][k] = (float) (Math.floor(buff[j][k] * 100.0) / 100.0);        // shave off all extra digits after the first two following the dot.
                    
                    if((buff[j][k] > 10)||(buff[j][k] < -10))
                    {
                        System.out.println("In was the weights initialisation!");
                    }
                }
            }
            
            // save the weights generated between these two layers
            weightsEmergingFromLayer.add(buff);
        }
        
    }
}
