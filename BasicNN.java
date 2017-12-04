package basicnn;

import java.util.Arrays;
import java.util.Random;

/*
 * @author tudorel
 */
public class BasicNN {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
     
        NeuralNetwork netw = new NeuralNetwork((float) 1, 1000_000, new int[]{2,6,10});
        
        int trainingSet=10;
        
        float[][][] trainingData = new float[trainingSet][2][];
        
        // we forcefully create an irregular matrix that has:
        // the array on [x][0][] of length 2
        // the array on [x][1][] of length 10
        for(int i=0; i<trainingSet; i++)
        {
            trainingData[i][0] = new float [2];
            trainingData[i][1] = new float [10];
        }
        
        
        for(int i=0; i<10; i++)
        {
            // training set: 1*0, 1*1, ..., 1*9
            // ----------------------------------------
            // input:
            trainingData[i][0][0] = 1;
            trainingData[i][0][1] = i;
            // output:
            // we have 10 possible output neurons; when the output should be 9, we expect neuron 9 to be close to 1
            trainingData[i][1][i] = 1;
            
            // we populate all the rest with 0
            for(int j=0; j<10; j++)
            {
                if(j != i)
                    {trainingData[i][1][j] = 0;}
            }
            // ----------------------------------------
            
            if(trainingSet > 10)
            {
            // training set: 2*0, 2*1, ..., 2*9
            // ----------------------------------------
            // input:
            trainingData[i+10][0][0] = 2;
            trainingData[i+10][0][1] = i;
            // output:
            // we populate all with 0
            for(int j=0; j<10; j++)
                {trainingData[i+10][1][j] = 0;}
            
            // we extract the output neurons mathematically by deducing:
            // if the result is > 10, then position 1 must be set
            if(2*i/10 == 1)
                {trainingData[i+10][1][1] = 1;}
            // the second digit to be set is the remainder of the division:
            trainingData[i+10][1][2*i%10] = 1;
            // ----------------------------------------
            
            
            // training set: 3*0, 3*1, ..., 3*9
            // ----------------------------------------
            // input:
            trainingData[i+20][0][0] = 3;
            trainingData[i+20][0][1] = i;
            // output:
            // we populate all with 0
            for(int j=0; j<10; j++)
                {trainingData[i+20][1][j] = 0;}
            
            // we extract the output neurons mathematically by deducing:
            // if the result is > 10, then position 1 must be set
            if(3*i/10 == 1)
                {trainingData[i+20][1][1] = 1;}
            else if(3*i/10 == 2)        // if we're > 20
                {trainingData[i+20][1][2] = 1;}
                
            // the second digit to be set is the remainder of the division:
            trainingData[i+20][1][3*i%10] = 1;
            // ----------------------------------------
            
            
            // training set: 4*0, 4*1, ..., 4*9
            // ----------------------------------------
            // input:
            trainingData[i+30][0][0] = 4;
            trainingData[i+30][0][1] = i;
            // output:
            // we populate all with 0
            for(int j=0; j<10; j++)
                {trainingData[i+30][1][j] = 0;}
            
            // we extract the output neurons mathematically by deducing:
            // if the result is > 10, then position 1 must be set
            if(4*i/10 == 1)
                {trainingData[i+30][1][1] = 1;}
            else if(4*i/10 == 2)        // if we're > 20
                {trainingData[i+30][1][2] = 1;}
            else if(4*i/10 == 3)        // if we're > 30
                {trainingData[i+30][1][3] = 1;}
                
            // the second digit to be set is the remainder of the division:
            trainingData[i+30][1][4*i%10] = 1;
            // ----------------------------------------
            
            
            // training set: 5*0, 5*1, ..., 5*9
            // ----------------------------------------
            // input:
            trainingData[i+40][0][0] = 5;
            trainingData[i+40][0][1] = i;
            // output:
            // we populate all with 0
            for(int j=0; j<10; j++)
                {trainingData[i+40][1][j] = 0;}
            
            // we extract the output neurons mathematically by deducing:
            // if the result is > 10, then position 1 must be set
            if(5*i/10 == 1)
                {trainingData[i+40][1][1] = 1;}
            else if(5*i/10 == 2)        // if we're > 20
                {trainingData[i+40][1][2] = 1;}
            else if(5*i/10 == 3)        // if we're > 30
                {trainingData[i+40][1][3] = 1;}
            else if(5*i/10 == 4)        // if we're > 40
                {trainingData[i+40][1][4] = 1;}
                
            // the second digit to be set is the remainder of the division:
            trainingData[i+40][1][5*i%10] = 1;
            // ----------------------------------------
            }
            
//            System.out.println("Input:  "+2+" and "+i+"; Output: "+Arrays.toString(trainingData[i+10][1]));
        }
        System.out.println("Started learning. ");
        netw.learn(trainingData);
        System.out.println("Done learning. ");
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("Tryout runs: ");
        
        System.out.println("Inputs 1 * [1.0 , 0.0 ... 9.0]");
        for(int j=0; j<10; j++)
        {
            float [] x = new float[]{1,j};
            float[] result = netw.run(x);
            int max = netw.determineMaxRes(result);
            System.out.println("Result: "+Arrays.toString(result)+". Position of max: "+max);
        }
        
//        System.out.println("Inputs 2 * [1.0 , 0.0 ... 9.0]");
//        for(int j=0; j<10; j++)
//        {
//            float [] x = new float[]{2,j};
//            float[] result = netw.run(x);
//            int[] max = netw.determineMaxResTwoDigits(result);
//            System.out.println("Result: "+Arrays.toString(result)+". Position of max: "+max[0]+max[1]);
//        }
        
//        float[] a = netw.sigmoid(-300);
//        System.out.println("-Sigmoid val: "+a[0]);
//        
//        a = netw.sigmoid(300);
//        System.out.println("+Sigmoid val: "+a[0]);
//        
//         // we fail because the sigmoidDerivative function can't process numbers smaller than -88
//        
//        a = netw.sigmoidDerivative(-45);     // old method
//        System.out.println("-Sigmoid derivative val old: "+a[0]);
//        
//        a = netw.sigmoidDerivative(1000);     // old method
//        System.out.println("+Sigmoid derivative val old: "+a[0]);

// ============================================================================================
// ============================================================================================
    }
    
    public static int determineMaxRes(double[] result) {
        double res=-1;
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

   

}
