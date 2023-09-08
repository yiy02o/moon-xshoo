import os
from subprocess import Popen, PIPE, STDOUT
import logging as log


def system_unzip(directory):
    count = 0
    for d in os.listdir(directory):
        filepath = os.path.join(directory, d)
        if os.path.isfile(filepath) and "fits" in d and os.path.splitext(filepath)[1] == ".Z":
            count += 1

    if count > 0:
        cmd = f"""uncompress {directory}/*fits.Z"""
        try:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            stdout, stderr = p.communicate()
            log.debug(f'output: {stdout}')
            print(f"Decompressed {count} fits.Z files")
        except Exception as e:
            log.error(f'Could not uncompress .Z files')


new_data_path = '/home/yiyo/Downloads/ESO_MOON_DATA_XSHOOTER/STD_TELL_SOLAR_TYPE/'
