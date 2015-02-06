import pandas as p
import numpy as np
import os
import math
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GMM


def get_statistics(data):
    time = 0.0
    distance = 0.0
    avg_x = np.mean(data[:, 0])
    avg_y = np.mean(data[:, 1])
    for idx in range(1, data.shape[0]):
        speed = math.sqrt((data[idx, 0] - data[idx - 1, 0])**2.0 + (data[idx, 1] - data[idx - 1, 1])**2.0)
        distance += speed
        time += 1.0
    avg_speed = distance / time

    return time, distance, avg_speed, avg_x, avg_y


def make_data():
    count = 0
    for subdir, dirs, _ in os.walk(os.path.join("data", "drivers")):
        for driver in dirs:
            count += 1
            if count % 10 == 0:
                print("at %d bitch" % count)
            drives = []
            drive_ids = []
            for subdir2, _, files in os.walk(os.path.join(subdir, driver)):
                for f in files:
                        drive_ids.append(f.split(".")[0])
                        data = p.read_csv(os.path.join(subdir2, f))
                        data = np.array(data)
                        t, d, s, x, y = get_statistics(data)
                        drives.append([t, d, s, x, y])
            drives = np.vstack(drives)
            np.savez(os.path.join("processed", driver), drives=drives, drive_ids=drive_ids)


def load_data():
    data = np.load(os.path.join("processed", "1" + ".npz"))
    drive_ids = data["drive_ids"]
    drives = data["drives"]


def train():
    submission = {"driver_trip": [], "prob": []} 

def main():
    submission = {"driver_trip": [], "prob": []} 
    count = 0
    for subdir, dirs, _ in os.walk(os.path.join("data", "drivers")):
        for driver in dirs:
            count += 1
            if count % 10 == 0:
                print("at %d bitch" % count)
            drives = []
            drive_ids = []
            for subdir2, _, files in os.walk(os.path.join(subdir, driver)):
                for f in files:
                        drive_ids.append(f.split(".")[0])
                        data = p.read_csv(os.path.join(subdir2, f))
                        data = np.array(data)
                        t, d, s = get_statistics(data)
                        drives.append([t, d, s])
            drives = np.vstack(drives)
            clf = GMM(n_components=2)
            clf.fit(drives)
            probs = clf.predict_proba(drives)
            if sum(probs[:, 0]) > sum(probs[:, 1]):
                probs = probs[:, 0]
            else:
                probs = probs[:, 1]
            # if sum(pred == 1) < sum(pred == 0):
            #     idx = 1
            #     pred[pred == 1] = -1
            #     pred[pred == 0] = 1
            #     pred[pred == -1] = 0

            for idx in range(len(drive_ids)):
                submission["driver_trip"].append("_".join([driver,
                                                           drive_ids[idx]]))
                submission["prob"].append(probs[idx])
    print("writing to file...")
    df = p.DataFrame(submission)
    df.to_csv("submission.csv", columns=["driver_trip", "prob"], index=0)
    return "stinky butthole"

if __name__ == "__main__":
    make_data()
    # load_data()
