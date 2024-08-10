import pandas as pd
import h5py
import sys
import capnp
import argparse
import numpy as np

# Define a function to read the .hits file
hit_capnp = capnp.load('/home/mayalaidler/seticore/hit.capnp')

def read_hits(filename):
    with open(filename, 'rb') as f:
        hits = hit_capnp.Hit.read_multiple(f)
        data = [hit.to_dict()['filterbank'] for hit in hits]
        f.seek(0,0)
        hits = hit_capnp.Hit.read_multiple(f)
        data2 = [hit.to_dict()['signal'] for hit in hits]
        d = pd.DataFrame(data)
        d2 = pd.DataFrame(data2)
        d3 = pd.concat([d, d2], axis = 1, join = 'outer')
    return d3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    # parser.add_argument('--bandwidth', help='Total bandwidth (Hz)',
    #                     type=float, required=True)
    # parser.add_argument('--nint', help='Number of time integrations',
    #                     type=int, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
   
    df = read_hits(args.input_file)
    
    for (i, row) in df.iterrows():
        
        data = np.array(row['data'])
    
        # Extract single scalar values
        num_time_steps = row['numTimesteps'][0] #if isinstance(row['numTimesteps'], pd.Series) else row['numTimesteps']
        num_channels = row['numChannels'] #if isinstance(row['numChannels'], pd.Series) else row['numChannels']
        print(num_time_steps)
        print(num_channels)
        
        # Reshape the data
        reshaped_data = data.reshape(num_time_steps, 1, num_channels)
        print(reshaped_data.shape)
    
        # x = np.fromfile(args.input_file, 'float32').reshape(-1, row['numChannels'])
        with h5py.File(args.output_file, 'w') as f_out:

            data = f_out.create_dataset('data', (reshaped_data.shape),
                                        dtype='f')
            data[:, :, :] = reshaped_data
            f_out.attrs['CLASS'] = b'FILTERBANK'
            f_out.attrs['VERSION'] = b'1.0'
            data.attrs['DIMENSION_LABELS'] = np.array(
                ['time', 'feed_id', 'frequency'], dtype='object')
            data.attrs['az_start'] = 0.0
            data.attrs['data_type'] = 1
            data.attrs['fch1'] = row['fch1']
            data.attrs['foff'] = row['foff']
            data.attrs['ibeam'] = row['beam'][0]
            data.attrs['machine_id'] = 0
            data.attrs['nbeams'] = 1
            data.attrs['nbits'] = 32
            data.attrs['nchans'] = row['numChannels']
            # number of fine channels per coarse channel
            # here we only have one coarse channel
            data.attrs['nfpc'] = row['numChannels']
            data.attrs['nifs'] = 1
            data.attrs['source_name'] = row['sourceName']
            data.attrs['src_dej'] = row['dec']
            data.attrs['src_raj'] = row['ra']
            data.attrs['telescope_id'] = 64
            data.attrs['tsamp'] = row['tsamp']
            data.attrs['tstart'] = row['tstart']
            data.attrs['za_start'] = 0.0
            
        print(f"Conversion complete. HDF5 file saved as {args.output_file}")


if __name__ == '__main__':
    main()
