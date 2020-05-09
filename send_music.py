import numpy as np
import serial
import time
wait_time = 0.1
# generate the waveform table
music_length = 42


# output formatter
formatter = lambda x: "%.3f" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(music_length * wait_time) * 2))
# first song
music = [0.261, 0.261, 0.392, 0.392, 0.440, 0.440, 0.392,
    0.349, 0.349, 0.330, 0.330, 0.294, 0.294, 0.261,
    0.392, 0.392, 0.349, 0.349, 0.330, 0.330, 0.294,
    0.392, 0.392, 0.349, 0.349, 0.330, 0.330, 0.294,
    0.261, 0.261, 0.392, 0.392, 0.440, 0.440, 0.392,
    0.349, 0.349, 0.330, 0.330, 0.294, 0.294, 0.261]
for data in music:
    s.write(bytes(formatter(data), 'UTF-8'))
    time.sleep(wait_time)
# second song
music = [0.100, 0.200, 0.300, 0.400, 0.500, 0.100, 0.200,
    0.300, 0.400, 0.500, 0.100, 0.200, 0.300, 0.400,
    0.500, 0.100, 0.200, 0.300, 0.400, 0.500, 0.100,
    0.200, 0.300, 0.400, 0.500, 0.100, 0.200, 0.300,
    0.400, 0.500, 0.100, 0.200, 0.300, 0.400, 0.500,
    0.100, 0.200, 0.300, 0.400, 0.500, 0.100, 0.200]
for data in music:
    s.write(bytes(formatter(data), 'UTF-8'))
    time.sleep(wait_time)
s.close()
print("Signal sended")