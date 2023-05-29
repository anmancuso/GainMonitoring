# GainMonitoring
Gain Monitoring using SavGol Filter
Description:


1. plugin_Area.py  = straxen plugin for LED calibration data type 
2. spe_analysis.py = fit functions for the Charge spectrum
3. execute_gain_plugin = This is the real code for the gain analysis. It creates the LED data type and saves the gain value for each run . Input: Run list
4. run_for_calibration_2.csv = Run database for calibration. Separates between Odd and Even column run.
5. test_filter notebook = Analysis of gain values, using the filter it creates a database of gain values evenly distributed in time. 
