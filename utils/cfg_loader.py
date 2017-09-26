
import os
def load_class_names(namesfile):
	class_names = []
	with open(namesfile,'r') as fp:
		lines = fp.readlines()
	for line in lines:
		line = line.rstrip()
		class_names.append(line)
	return class_names


def read_data_cfg(datacfg):
    options = dict()
    with open(datacfg,'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if len(line)==0:
            return options
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


