#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob
import json
import uuid
from datetime import datetime, timezone

from SyncNetInstance import *

def main(opt, filename=None):
    # ==================== LOAD MODEL AND FILE LIST ====================

    s = SyncNetInstance()

    s.loadParameters(opt.initial_model)
    print("Model %s loaded." % opt.initial_model)

    flist = glob.glob(os.path.join(opt.crop_dir, opt.reference, '0*.avi'))
    flist.sort()
    
    conscent_video_info = {
        "videoFileName": opt.videofile,
        "timeStamp": datetime.now(timezone.utc).strftime("UTC-0: %Y-%m-%d-%H-%M-%S"),
        "id": uuid.uuid4().hex,
        "status": False,
        "statusMessage": "No processed video detected. Please check the input video whether it meets the requirements, e.g. the face can be always detected in each video frame.",
    }

    if len(flist) == 0:
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(json.dumps(conscent_video_info, indent=4))
            f.close()
        return conscent_video_info

    # ==================== GET OFFSETS ====================

    dists = []
    for idx, fname in enumerate(flist):
        try:
            offset, conf, dist, min_dist = s.evaluate(opt, videofile=fname)
            dists.append(dist)

            conscent_video_info.update({
                "videoFileName": opt.videofile,
                "timeStamp": datetime.now(timezone.utc).strftime("UTC-0: %Y-%m-%d-%H-%M-%S"),
                "id": uuid.uuid4().hex,
                "avOffset": float(offset),
                "minDist": float(min_dist),
                "confidenceScore": float(conf),
                "status": True,
                "statusMessage": "Success"
            })
        except Exception as e:
            conscent_video_info.update({
                "status": False,
                "statusMessage": f"Can't get lipsync scores using SyncNet."
            })

        if filename is not None:            
            # with open(filename, 'a') as f:
            #     f.write("%s %f %f %f %f %f %f\n" % (opt.videofile, offset, conf, dist.min(), dist.max(), dist.mean(), numpy.median(dist)))
            # f.close()
            with open(filename, 'w') as f:
                f.write(json.dumps(conscent_video_info, indent=4))
            f.close()
            
        else:
            conscent_video_info.update({
                "status": False,
                "statusMessage": f"No such file {filename} provided, cannot write to it."
            })
            return conscent_video_info

        
    # ==================== PRINT RESULTS TO FILE ====================

    with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
        pickle.dump(dists, fil)
    
    
    return conscent_video_info



if __name__ == '__main__':

    # ==================== PARSE ARGUMENT ====================

    parser = argparse.ArgumentParser(description = "SyncNet")
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--data_dir', type=str, default='data/work', help='')
    parser.add_argument('--videofile', type=str, default='', help='')
    parser.add_argument('--reference', type=str, default='', help='')
    parser.add_argument('--save_file_path', type=str, default='./result.json', help='')
    opt = parser.parse_args()

    setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
    setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
    setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
    setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
    
    main(opt=opt, filename=opt.save_file_path)