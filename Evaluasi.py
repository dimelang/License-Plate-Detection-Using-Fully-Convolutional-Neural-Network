# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:52:38 2020

@author: Dim
"""
from shapely.geometry import Polygon
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt


def checkIntersection(tar, pred):
    xs = np.ravel(tar[:, :, 0].T)
    ys = np.ravel(tar[:, :, 1].T)

    xs1 = np.ravel(pred[:, :, 0].T)
    ys1 = np.ravel(pred[:, :, 1].T)

    a = []
    for i in range(len(xs)):
        a.append([xs[i], ys[i]])

    b = []
    for i in range(len(xs1)):
        b.append([xs1[i], ys1[i]])

    p1 = Polygon(a)
    p2 = Polygon(b)
    return p1.intersects(p2)


def Iou(tar, pred):
    target = tar
    xs = np.ravel(target[:, :, 0].T)
    ys = np.ravel(target[:, :, 1].T)

    prediksi = pred
    xs1 = np.ravel(prediksi[:, :, 0].T)
    ys1 = np.ravel(prediksi[:, :, 1].T)

    xA = max(xs1[0], xs[0])
    yA = max(ys1[0], ys[0])

    xB = min(xs1[2], xs[2])
    yB = min(ys1[2], ys[2])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (xs[2] - xs[0] + 1) * (ys[2] - ys[0] + 1)
    boxBArea = (xs1[2] - xs1[0] + 1) * (ys1[2] - ys1[0] + 1)
    if float(boxAArea + boxBArea - interArea) == 0.0:
        return 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


def getTrueLabel(label, size):
    label = label[:-1]
    newSize = size
    label = (newSize/960)*label
    KiA_KaB = [label[::2, :, 1], label[::2, :, 0]]
    KaA_KiB = [label[1::2, :, 1], label[1::2, :, 0]]

    KiA_KaB = np.array(KiA_KaB).T
    KaA_KiB = np.array(KaA_KiB).T
    gabung = []

    for i in range(KiA_KaB.shape[0]):
        gabung.append(np.concatenate([KiA_KaB[i], KaA_KiB[i]]))
    gabung = np.reshape(gabung, newshape=(
        KaA_KiB.shape[0], KaA_KiB.shape[1], KaA_KiB.shape[2], KaA_KiB.shape[2]))
    return np.round(gabung).astype(int)


def localMaxima(heatmap, distance):
    return peak_local_max(heatmap, min_distance=distance, threshold_rel=heatmap.max()*0.5, num_peaks=4)


def extractKey(c, K1, K2):
    jarak = []
    min_jarak = []
    K1_K2 = []
    keypoint_K1_K2 = []

    for i in K1:
        for j in K2:
            cKey = (i+j)/2
            dist = np.sqrt(np.sum(np.square(c-cKey)))
            jarak.append(dist)
            K1_K2.append([i, j])

    jarak = np.array(jarak)
    K1_K2 = np.array(K1_K2)
    index = np.where(jarak == jarak.min())
    tempMin_jarak = jarak[index]
    key = np.array(K1_K2[index])

    if len(key) > 1:
        key = np.delete(key, slice(0, -1, 1), axis=0)
        tempMin_jarak = np.delete(tempMin_jarak, slice(0, -1, 1), axis=0)
    min_jarak.append(tempMin_jarak)
    min_jarak = np.array(min_jarak)
    keypoint_K1_K2.append([key[0, 0], key[0, 1]])
    return keypoint_K1_K2, min_jarak


def checkDuplicate(keyPoint, jarak):
    index = 0
    temp = []
    for i in range(1, keyPoint.shape[0]):
        if i-1 == 0:
            temp.append(keyPoint[index])
        kondisi = temp[index] in keyPoint[i]
        if kondisi == True:
            min = [np.sum(jarak[index]), np.sum(jarak[i])]
            indices = np.where(min == np.array(min).min())
            indices = indices[0]
            if indices[0] == 1:
                temp[index] = keyPoint[i]
            elif indices[0] == 0:
                temp[index] = temp[index]
        elif kondisi == False:
            temp.append(keyPoint[i])
            index += 1
    return temp


def KeypointPrediction(heatmap):
    im = heatmap
    keyPoint = []
    jarak = []

    coordinates = [localMaxima(im[:, :, i], 1)for i in range(im.shape[-1])]
    countZero = 0
    for i in coordinates:
        if np.array(i).size == 0:
            countZero += 1
        else:
            countZero = countZero

    if countZero > 0:
        return keyPoint
    else:
        coordinates = np.array(coordinates)
        for x, c in enumerate(coordinates[-1]):
            keyPoint.append([])
            jarak.append([])
            KiA_KaB, jarak_KiA_KaB = extractKey(
                c, coordinates[0], coordinates[2])
            KaA_KiB, jarak_KaA_KiB = extractKey(
                c, coordinates[1], coordinates[3])
            keyPoint[x].append([KiA_KaB, KaA_KiB])
            jarak[x].append([jarak_KiA_KaB, jarak_KaA_KiB])
        keyPoint = np.reshape(keyPoint, newshape=(np.array(keyPoint).shape[0], np.array(
            keyPoint).shape[2], np.array(keyPoint).shape[4], np.array(keyPoint).shape[5]))
        if keyPoint.shape[0] > 1:
            keyPoint = checkDuplicate(keyPoint, jarak)
            return np.array(keyPoint)
        else:
            return keyPoint


def confusionMatrix2(keypointTar, keypointPred, thresIou):
    check = []
    TP, FP, FN = (0, 0, 0)
    if np.array(keypointPred).size == 0:
        FN += keypointTar.shape[0]
    else:
        for i in range(keypointTar.shape[0]):
            check.append([])
            for j in range(keypointPred.shape[0]):
                kondisiInter = checkIntersection(
                    keypointTar[i], keypointPred[j])
                kondisiIou = Iou(keypointTar[i], keypointPred[j])
                if kondisiInter == True and kondisiIou >= thresIou:
                    check[i].append(True)
                elif kondisiInter == False or kondisiIou < thresIou:
                    check[i].append(False)
        FP += keypointPred.shape[0] - np.count_nonzero(check)
        TP += np.count_nonzero(check)
        if keypointPred.shape[0] < keypointTar.shape[0]:
            FN += (keypointTar.shape[0]-keypointPred.shape[0])
    return TP, FP, FN


def mse(pred, tar):
    loss = []
    for i in range(pred.shape[-1]):
        err = np.sum((pred[:, :, i].astype("float") -
                      tar[:, :, i].astype("float")) ** 2)
        err /= float(pred.shape[0] * pred.shape[1])
        loss.append(err)
    return np.mean(loss)


def AP(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)

    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii


def evaluate(pred, target, label, threshIou, sizeImage):
    confMat = []
    Loss = []
    for x in range(len(pred)):
        Loss.append(mse(pred[x], target[x]))

    for i in range(len(pred)):
        index = i
        test = np.array(pred[index])
        im = test
        heatmapLabel = label[index]
        keypointPred = KeypointPrediction(im)
        keypointTar = getTrueLabel(heatmapLabel, sizeImage)
        confusionMat = confusionMatrix2(keypointTar, keypointPred, threshIou)
        confMat.append(confusionMat)

    confMat = np.array(confMat)

    temp = []
    for i in confMat[:, :2]:
        tp = i[0]
        fp = i[1]
        for x in range(tp):
            temp.append([1, 0])
        for y in range(fp):
            temp.append([0, 1])

    temp = np.array(temp)

    ACCconfMat = np.cumsum(temp, axis=0)
    count = [label[i].shape[1]for i in range(len(label))]
    count = np.sum(count)

    precision = [ACCconfMat[i, 0]/(ACCconfMat[i, 0]+ACCconfMat[i, 1]) if (ACCconfMat[i, 0]+ACCconfMat[i, 1])
                 != 0 else 0.0 for i in range(len(ACCconfMat))]  # untuk handle pembagi nol
    recall = [ACCconfMat[i, 0]/count for i in range(len(ACCconfMat))]

    averagePrecision = AP(recall, precision)[0]

    print("AP50:", "{:.4%}".format(averagePrecision))
    print("Loss :", np.mean(Loss))
    return np.mean(Loss), averagePrecision
