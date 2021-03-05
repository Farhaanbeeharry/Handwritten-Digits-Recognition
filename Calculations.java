/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package coursework2;

/**
 *
 * @author Farhaan Beeharry
 */
public class Calculations {
    
    public static double roundOff2DP(double number) {
        double roundedValue = Math.round(number * 100.0) / 100.0;
        return roundedValue;
    }
    
}
