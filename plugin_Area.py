import datetime
from immutabledict import immutabledict
import strax
import numpy as np

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()

#Channel List of nVeto PMTs
#Eventual Masked channels have to be added in a config file.
channel_list=[i for i in range(2000,2120)]


@export
@strax.takes_config(
    strax.Option('baseline_window_nv',
                 default=(40,90),
                 help="Window (samples) for baseline calculation."),
    strax.Option('led_window_nv',
                 default=(100, 140),
                 help="Window (samples) where we expect the signal in LED calibration"),
    strax.Option('integration_window_nv',
                 default=(5, 10),
                 help="Integration window [-x,+y] from the peak"),
    strax.Option('channel_list_nv',
                 default=(tuple(channel_list)), infer_type=False,
                 help="List of PMTs. Defalt value: all the PMTs"),
    strax.Option('acq_window_length_nv',
                 default=160,
                 help="Length of the Acq. Win. (samples). Defalt value: 320 samples")
    )




class NVLEDCalibration(strax.Plugin):
    """
        Preliminary version.
        LEDCalibration returns: channel, time, endtime , Area, Baseline (RMS) ,Amplitude and Peak position..
        The new variables are:
            - Area: Area computed in the given window. The integration is performed defining
            a dinamic window from the peak [ADC Counts x Samples].
            - Amplitude LED: peak amplitude of the LED on run in the given
            window [ADC Counts].
            - signal_time: position of the Peak wrt to the trigger time [Samples].
    """

    __version__ = '0.0.1'
    depends_on = ('raw_records_nv')
    provides = 'led_cal_nv'
    data_kind = 'hitlets_nv'
    
    dtype = strax.time_fields + [
                (('Channel/PMT number', 'channel'),np.int16),
                (('Baseline in the given window', 'baseline'),np.float32),
                (('Baseline error in the given window', 'baseline_rms'), np.float32),
                (('Amplitude in the given window', 'amplitude'),np.float32),
                (('Sample/index of amplitude in the given window', 'signal_time'), np.float32),
                (('Integrated charge in a the given window', 'area'), np.float32)     
            ]
    
    
    def compute(self,raw_records_nv):
        #mask = np.where(np.in1d(raw_records_nv["channel"],self.config["channel_list"]))[0]
        
        
        temp = np.zeros(len(raw_records_nv[raw_records_nv["record_i"]==0]), dtype=self.dtype)
        channels=self.config['channel_list_nv']
        length=self.config["acq_window_length_nv"]
        window_bsl=self.config["baseline_window_nv"]
        window=self.config["led_window_nv"]
        int_win=self.config["integration_window_nv"]
        
        temp=self.SPE_waveform_processing(raw_records_nv,length,channels,window_bsl,window,int_win)
        del raw_records_nv
        return temp
    
    @staticmethod
    def SPE_waveform_processing(rr,length,channels,window_bsl,window,int_win):
        
        mask_record_0=rr["record_i"]==0
        record0=rr[mask_record_0]
        mask_record_1=rr["record_i"]==1
        record1=rr[mask_record_1]

        if channels != None:
            mask0           = np.where(np.in1d(record0['channel'], channels))[0]
            mask1           = np.where(np.in1d(record1['channel'], channels))[0]
            
            _raw_records0   = record0[mask0]
            _raw_records1   = record1[mask1]
        else:
            _raw_records0 = record0
            _raw_records1 = record1
            
            
        _dtype = strax.time_fields + [
                (('Channel/PMT number', 'channel'),np.int16),
                (('Baseline in the given window', 'baseline'),np.float32),
                (('Baseline error in the given window', 'baseline_rms'), np.float32),
                (('Amplitude in the given window', 'amplitude'),np.float32),
                (('Sample/index of amplitude in the given window', 'signal_time'), np.float32),
                (('Integrated charge in a the given window', 'area'), np.float32)     
            ]    
            
            
        res = np.zeros(len(_raw_records0), dtype = _dtype)
        
        
        #Merging Waveform
        waveforms = np.hstack((_raw_records0["data"],_raw_records1["data"][:,0:length-110]))
        #Baseline
        baseline = np.mean(waveforms[:,window_bsl[0]:window_bsl[1]],axis=1)
        baseline_std = np.std(waveforms[:,window_bsl[0]:window_bsl[1]],axis=1)
        #Reverse Waveform
        waveforms = (-1)*(waveforms-baseline[:,None])
        #Amplitude
        amplitude = np.max(waveforms[:,window[0]:window[1]],axis=1)
        signal_time = np.argmax(waveforms[:,window[0]:window[1]],axis=1)+window[0]
        #Integration around Peak 
        low_int = signal_time-int_win[0]
        up_int = signal_time+int_win[1]
        col = np.array([np.arange(0,length) for i in range(len(waveforms))])
        tmask = (low_int[:,None] <= col) & (col < up_int[:,None])
        area = np.where(tmask, waveforms,0).sum(axis=1)
        
        
        res['channel'] = _raw_records0['channel']
        res['time'] = _raw_records0['time']
        res['endtime']=_raw_records0['pulse_length']*_raw_records0['dt']+_raw_records0['time']
        res['baseline'] = baseline
        res['baseline_rms'] = baseline_std
        res['amplitude'] = amplitude
        res['signal_time'] = signal_time
        res['area'] = area 
        
        return res
        
        