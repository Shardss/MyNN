package basicnn;

import java.util.Arrays;

/*
 * @author tudorel
 */
public class BasicNN {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
     
        NeuralNetwork netw = new NeuralNetwork((float) 1, 1500000, new int[]{2,6,10});
        
        float[][][] trainingData = new float[50][2][];
        
        // we forcefully create an irregular matrix that has:
        // the array on [x][0][] of length 2
        // the array on [x][1][] of length 10
        for(int i=0; i<50; i++)
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
            
            
            // training set: 4*0, 4*1, ..., 4*9
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
        
        System.out.println("Inputs 2 * [1.0 , 0.0 ... 9.0]");
        for(int j=0; j<10; j++)
        {
            float [] x = new float[]{2,j};
            float[] result = netw.run(x);
            int[] max = netw.determineMaxResTwoDigits(result);
            System.out.println("Result: "+Arrays.toString(result)+". Position of max: "+max[0]+max[1]);
        }
        
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
//[Run: 3760000; input: 1 & 0; calculated output: 0 errors: [0.0, -2.8625186E-20, -1.0, -1.6628991E-14, -6.30866E-14, -7.6156127E-31, -2.8625186E-20, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 1; calculated output: 0 errors: [-1.0, 1.0, -1.0, -5.0075084E-16, -1.958271E-15, -3.1872868E-34, -2.8625186E-20, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 2; calculated output: 0 errors: [-1.0, -2.8625186E-20, 0.0, -2.256682E-16, -8.886282E-16, -5.4288536E-35, -2.8625186E-20, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 3; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, 1.0, -7.696071E-16, -3.9337454E-35, -2.8625186E-20, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 4; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, -1.9033059E-16, 1.0, -3.7192558E-35, -2.8625186E-20, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 5; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, -1.895018E-16, -7.4734146E-16, 1.0, -2.8625186E-20, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 6; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, -1.8935944E-16, -7.467886E-16, -3.6772436E-35, 1.0, -1.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 7; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, -1.8933488E-16, -7.466918E-16, -3.6762056E-35, -2.8625186E-20, 0.0, -2.8625186E-20, -2.8625186E-20]
// Run: 3760000; input: 1 & 8; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, -1.8933055E-16, -7.466747E-16, -3.6760093E-35, -2.8625186E-20, -1.0, 1.0, -2.8625186E-20]
// Run: 3760000; input: 1 & 9; calculated output: 0 errors: [-1.0, -2.8625186E-20, -1.0, -1.8932982E-16, -7.466718E-16, -3.6760093E-35, -2.8625186E-20, -1.0, -2.8625186E-20, 1.0]


//double[] caca0 = new double []{7.748604E-6, -0.005226875, -2.8625186E-20, -2.8625186E-20, -2.8625186E-20, -2.8625186E-20, -2.8625186E-20, -2.8625186E-20, -2.8625186E-20, -7.4717116E-7};
//double[] caca1 = new double []{-1.2273242E-5, 0.019875646, -0.06785322, -4.1624506E-18, -2.8625186E-20, -2.2211866E-24, -2.5616053E-18, -2.8993747E-15, -1.1376481E-12, -1.3232975E-4};
//double[] caca2 = new double []{-6.6773566E-7, -0.006828959, 0.8363198, -0.07553027, -0.01190422, -0.068686225, -0.07997582, -0.07954458, -0.07741444, -2.6965674E-4};
//double[] caca3 = new double []{-6.0014804E-7, -0.0044101635, -0.19530742, 0.76633483, -0.35038486, -0.34500885, -0.26916727, -0.21871427, -0.17757885, -2.761308E-4};
//double[] caca4 = new double []{-5.979109E-7, -0.0043086573, -0.12918109, -0.36033857, 0.7918048, -0.22907765, -0.2065935, -0.18194923, -0.15634356, -2.7341078E-4};
//double[] caca5 = new double []{-5.9764636E-7, -0.004193774, -0.04562428, -0.046953566, -0.28827512, 0.87806344, -0.14254257, -0.13919899, -0.12791725, -2.6288469E-4};
//double[] caca6 = new double []{-5.969577E-7, -0.0038122532, -0.0013841426, -2.0152E-4, -0.003968881, -0.15608233, 0.91028035, -0.105636515, -0.10430751, -2.2802809E-4};
//double[] caca7 = new double []{-5.9451185E-7, -0.0027117673, -4.8883466E-9, -6.5019236E-13, -6.3458945E-9, -0.0022337025, -0.1359644, 0.9276884, -0.0837379, -1.3710458E-4};
//double[] caca8 = new double []{-5.862104E-7, -8.435521E-4, -1.0338055E-27, -2.8625186E-20, -8.822352E-29, -2.2940299E-9, -0.004463597, -0.12739927, 0.9392759, -2.4003746E-5};
//double[] caca9 = new double []{-5.6219443E-7, -2.6039124E-5, -2.8625186E-20, -2.8625186E-20, -2.8625186E-20, -3.496505E-27, -3.3177074E-7, -0.009798073, -0.12831298, 0.9999999};
//
//System.out.println(determineMaxRes(caca0));
//System.out.println(determineMaxRes(caca1));
//System.out.println(determineMaxRes(caca2));
//System.out.println(determineMaxRes(caca3));
//System.out.println(determineMaxRes(caca4));
//System.out.println(determineMaxRes(caca5));
//System.out.println(determineMaxRes(caca6));
//System.out.println(determineMaxRes(caca7));
//System.out.println(determineMaxRes(caca8));
//System.out.println(determineMaxRes(caca9));
    


    System.out.println();
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
