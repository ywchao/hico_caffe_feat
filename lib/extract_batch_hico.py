#!/usr/bin/env python

from extract import Extractor

import numpy as np
import os
import sys
import argparse
import time
import fnmatch
from scipy import io

import caffe


def get_process_file(ip_folder, op_folder, num_batch, batch_id):
    # set image sets
    im_sets = ['train2015', 'test2015']

    # set input/output directories
    ip_dirs = [ip_folder + '/' + im_set for im_set in im_sets]
    op_dir = op_folder
    print 'input dirs'
    for ip_dir in ip_dirs:
        print ip_dir
    print 'output_dir'
    print op_dir

    # make output directory
    os.system('mkdir -p {}'.format(op_dir))

    # print number of images
    print 'number of images'
    for im_set, ip_dir in zip(im_sets, ip_dirs):
        print '{:11} {:>5}'.format(im_set + ':', len(os.listdir(ip_dir)))
    total = sum([len(os.listdir(ip_dir)) for ip_dir in ip_dirs])
    print '{:11} {:>5}'.format('total: ', total)
    
    # get start and end id
    interval = round(total / num_batch);
    sid = interval * (batch_id -1) + 1;
    eid = interval * batch_id;
    if batch_id == num_batch:
        eid = total
    print '{:11} {:>5}'.format('process: ', int(eid - sid + 1))

    # get batch_src and batch_des
    cnt = 0
    list_src = []
    list_des = []
    for ip_dir in ip_dirs:
        list_fname = os.listdir(ip_dir)
        list_fname.sort()
        for fname in list_fname:
            cnt += 1

            if cnt < sid:
                continue
            if cnt > eid:
                break

            fname_mat = fname.rpartition('.')[0] + '.mat'
            list_src.append(os.path.join(ip_dir, fname))
            list_des.append(os.path.join(op_dir, fname_mat))

    return list_src, list_des

def main(argv):
    caff_root = 'caffe'
    mypycaffe_dir = os.path.join(caff_root, 'python')

    parser = argparse.ArgumentParser()
    # Required arguments: input file path;
    parser.add_argument(
        "input_folder",
        help="HICO image folder containing 'train2015' and 'test2015'."
    )
    parser.add_argument(
        "output_folder",
        help="Folder to save output features."
    )
    parser.add_argument(
        "num_batch",
        type=int,
        help="Number of batches."
    )
    parser.add_argument(
        "batch_id",
        type=int,
        help="Batch index."
    )
    # Optional arguments.
    parser.add_argument(
        "--chunk_size",
        default=10,
        type=int,
        help="Number of images to work on at one time."
    )
    parser.add_argument(
        "--model_def",
        default=os.path.join(mypycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(mypycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    # parser.add_argument(
    #     "--center_only",
    #     action='store_true',
    #     help="Switch for prediction from center crop alone instead of " +
    #          "averaging predictions across crops (default)."
    # )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(mypycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    # new arguments
    parser.add_argument(
        "--crop_mode",
        default='oversample',
        help="Set the mode for cropping input images."
    )

    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        # mean = np.load(args.mean_file).mean(1).mean(1)
        if len(args.mean_file) > 8 and args.mean_file[:8] == 'setmean-':
            if args.mean_file[8:] == 'VGG16':
                mean = np.array([102.9801, 115.9465, 122.7717])
            # Add more cases here.
        else:
            mean = np.load(args.mean_file).mean(1).mean(1)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    net = Extractor(args.model_def, args.pretrained_model,
                    image_dims=image_dims, mean=mean,
                    input_scale=args.input_scale, 
                    raw_scale=args.raw_scale,
                    channel_swap=channel_swap,
                    feature_name="fc7")

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # get list_src and list_des
    list_src, list_des = get_process_file( \
        args.input_folder, args.output_folder, args.num_batch, args.batch_id)

    # get chunk size
    chunk_size = args.chunk_size 

    # start extract
    cnt = 0
    current = 0
    chunk_src = []
    chunk_des = []
    for src, des in zip(list_src, list_des):
        # skip if output file exists
        try:
            garbage = io.loadmat(des)
            cnt += 1
            print '{:05d}/{:05d} {}'.format(cnt,len(list_src), \
                                            os.path.basename(src))
            continue
        except:
            # print dest
            pass

        # start batch
        if current == 0:
           print 'start chunk'

        # update cnt and current
        cnt += 1
        current += 1
        chunk_src.append(src)
        chunk_des.append(des)
        print '{:05d}/{:05d} {}'.format(cnt,len(list_src), \
                                        os.path.basename(src))

        # process batch
        if current == chunk_size or cnt == len(list_src):
            # load image
            try:
                inputs = [caffe.io.load_image(img_f) for img_f in chunk_src]
            except IOError as e:
                print "I/O error: " + str(e)
                current = 0
                chunk_src = []
                chunk_des = []
                continue
            except ValueError as e:
                print "value error: " + str(e)
                current = 0
                chunk_src = []
                chunk_des = []
                continue

            # extract feature
            # features = net.extract(inputs)
            features = net.extract(inputs, args.crop_mode)

            # save feature
            for index, feature in enumerate(features):
                io.savemat(chunk_des[index], {'feat': feature})

            # reset
            current = 0
            chunk_src = []
            chunk_des = []

            print "chunk done: processed {} images.".format(cnt)

    print 'done.'

if __name__ == '__main__':
    main(sys.argv)

