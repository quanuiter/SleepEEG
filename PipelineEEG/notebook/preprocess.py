import argparse
import glob
import ntpath
import os
import shutil
import pyedflib
import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, lfilter

import logging

# Định nghĩa trực tiếp từ điển nhãn (thay cho sleepstage.py)
stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "MOVE": 5,
    "UNK": 6,
}

# Định nghĩa trực tiếp hàm ghi log (thay cho logger.py)
def get_logger(log_file, level="info"):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # Lưu log vào file
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # In log ra màn hình console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}

# ========================================================================
# [NEW] SIGNAL PROCESSING FUNCTIONS
# ========================================================================

def bandpass_filter(signal, fs, low=0.5, high=30.0, order=4):
    """Butterworth bandpass 0.5–30 Hz to remove slow drifts and high-frequency noise."""
    sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, signal)

def notch_filter(signal, fs, freq=50.0, Q=30.0):
    """Notch filter at 50 Hz to remove powerline noise."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return lfilter(b, a, signal)

def clip_and_scale(signal, clip_uv=800.0):
    """Clip artifact ±800µV và scale bằng hằng số để GIỮ NGUYÊN tỷ lệ biên độ."""
    signal = np.clip(signal, -clip_uv, clip_uv)
    
    # Thay vì Z-score làm biến dạng biên độ, ta chia cho 100 
    # (vì tín hiệu EEG thường dao động trong khoảng ±100µV)
    # Lúc này dữ liệu sẽ nằm loanh quanh trong khoảng [-8.0, 8.0]
    signal = signal / 100.0 
    
    return signal.astype(np.float32)

# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/sleepedf/sleep-cassette",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="./Fpz-Cz",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG Fpz-Cz",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    raw_psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    raw_ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    
    raw_psg_fnames.sort()
    
    psg_fnames = []
    ann_fnames = []
    
    for psg_file in raw_psg_fnames:
        basename = os.path.basename(psg_file)
        # Lấy 7 ký tự đầu làm mã định danh chung (VD: SC4061E)
        subject_id = basename[:7]
        
        # Tìm file Hypnogram có chung mã định danh
        matching_ann = [f for f in raw_ann_fnames if os.path.basename(f).startswith(subject_id)]
        
        if len(matching_ann) > 0:
            psg_fnames.append(psg_file)
            ann_fnames.append(matching_ann[0])
        else:
            logger.warning(f"Bỏ qua: Không tìm thấy file nhãn cho tín hiệu {basename}")
            
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = psg_f.datarecord_duration
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        select_ch_idx = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch:
                select_ch_idx = s
                break
        if select_ch_idx == -1:
            raise Exception("Channel not found.")
        
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
        n_epoch_samples = int(epoch_duration * sampling_rate)
        
        # ==========================================================
        # [MODIFIED] SIGNAL FILTERING & NORMALIZATION
        # ==========================================================
        
        # 1. Read the ENTIRE raw signal first
        raw_signal = psg_f.readSignal(select_ch_idx)
        
        # 2. Apply filters to the entire continuous recording (avoids edge artifacts)
        raw_signal = bandpass_filter(raw_signal, fs=sampling_rate)
        raw_signal = notch_filter(raw_signal, fs=sampling_rate)
        
        # 3. Reshape into epochs
        signals = raw_signal.reshape(-1, n_epoch_samples)
        
        # 4. Apply clipping and scaling PER EPOCH
        signals = np.stack([clip_and_scale(s) for s in signals])
        
        # ==========================================================
        
        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
        logger.info("Sample rate: {}".format(sampling_rate))

        # Sanity check
        n_epochs = psg_f.datarecords_in_file
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels (Fixed np.int deprecation)
            label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x, 
            "y": y, 
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": len(x),
            "filtered": True,                  # Track that data is filtered
            "filter_range": [0.5, 30.0],       # Track filter parameters
            "normalization": "scale_100",      # Đánh dấu đã dùng scale thay vì z-score
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")

if __name__ == "__main__":
    main()