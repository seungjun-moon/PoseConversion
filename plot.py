import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.load import load_pickle

def main(args):
	os.makedirs(args.save_path, exist_ok=True)

	pose, cam, exp, shape = load_pickle(args.load_path)

	for i in range(pose.shape[1]):
		x = pose[:,i]
		plt.plot(x)
		plt.savefig(os.path.join(args.save_path, 'pose_{}.png'.format(str(i).zfill(4))))
		plt.cla()

	for i in range(exp.shape[1]):
	# for i in range(10):
		x = exp[:,i]
		
		plt.plot(x)
		plt.savefig(os.path.join(args.save_path, 'exp_{}.png'.format(str(i).zfill(4))))
		plt.cla()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='/home/june1212/PoseConversion/examples/flame_next3d.pkl', help='original pose path')
    parser.add_argument('--save_path', type=str, default='/home/june1212/PoseConversion/plot', help='save path')
    args = parser.parse_args()
    main(args)