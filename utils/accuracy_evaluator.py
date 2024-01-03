from collections import defaultdict
import numpy as np


def calculate_iou(pred_box, gt_box):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    :param pred_box: the predicted bounding box, list of 4 numbers [x_min, y_min, x_max, y_max]
    :param gt_box: the ground truth bounding box, list of 4 numbers [x_min, y_min, x_max, y_max]
    :return: the IoU score between the two boxes, scalar
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    boxBArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def calculate_ap(precision, recall):
    """
    Calculate the Average Precision (AP) given precision and recall values.
    :param precision: the precision values, np array of shape (num_pred_boxes, )
    :param recall: the recall values, np array of shape (num_pred_boxes, )
    :return: the AP score, scalar
    """
    # method of calculation follows PASCAL VOC 2012
    # adding two more points, representing minimum recall and maximum recall
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    # make the precision monotonically decreasing by iterating and
    # comparing with the next precision value
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the points where recall changes
    changing_points = np.where(recall[:-1] != recall[1:])[0]

    # sum up the different areas to calculate AP
    ap = np.sum((recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1])

    assert ap <= 1, "AP cannot be greater than 1."

    return ap


def calculate_accuracy(all_predictions, all_targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate the mean Average Precision (mAP) for given predictions and ground truths.
    :param all_predictions: the predictions of the model, list of dicts like
                            [{"boxes": torch.FloatTensor, "labels": torch.IntTensor, "scores": torch.FloatTensor}]
    :param all_targets: the ground truth of the model, list of dicts like
                        [{"boxes": torch.FloatTensor, "labels": torch.IntTensor}]
    :param iou_threshold: the threshold of IoU, scalar
    :param score_threshold: the threshold of score, scalar
    :return: the mAP score, scalar
    """
    # store AP values for each class
    ap_per_class = defaultdict(list)

    # iterate through each image
    for preds, targets in zip(all_predictions, all_targets):
        # process each class
        for class_id in range(len(preds["labels"])):
            pred_scores = preds["scores"][preds["labels"] == class_id]  # scores of the predicted boxes
            # filter out the predictions in score order and score greater than the threshold
            pred_scores, sort_ind = pred_scores.sort(descending=True)

            pred_boxes = preds["boxes"][preds["labels"] == class_id][sort_ind]
            pred_boxes = pred_boxes[pred_scores >= score_threshold]  # predicted boxes

            gt_boxes = targets["boxes"][targets["labels"] == class_id]  # ground truth boxes

            # if no prediction or ground truth, skip
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            # calculate IoU and metrics for each predicted box
            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))
            for i, pred_box in enumerate(pred_boxes):  # iterate through each predicted box
                ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0  # highest IoU, i.e. best match

                # if the highest IoU is above the threshold, count as true positive, else false positive
                if max_iou >= iou_threshold:
                    tp[i] = 1
                else:
                    fp[i] = 1

            # accumulate and compute precision and recall
            acc_fp = np.cumsum(fp)  # accumulated false positive (e.g.[0, 1, 1, 2, 2, 2, 3, 3, 4, 5, ...])
            acc_tp = np.cumsum(tp)  # accumulated true positive
            recall = acc_tp / len(gt_boxes)
            assert (recall <= 1).all(), "Recall cannot be greater than 1."
            precision = np.divide(acc_tp, (acc_fp + acc_tp))

            # calculate and store AP
            ap = calculate_ap(precision, recall)
            ap_per_class[class_id].append(ap)

    # calculate mean of all APs for mAP
    mAP = np.mean([np.mean(ap) for ap in ap_per_class.values()])
    return mAP
