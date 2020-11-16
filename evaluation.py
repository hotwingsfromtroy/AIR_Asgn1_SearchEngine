import json
import statistics

with open("results.json","r") as json_file:
    our_results = json.load(json_file)

with open("es_results.json","r") as json_file:
    es_results = json.load(json_file)

our_ft = our_results['freetext']
our_pt = our_results['phrase']
es_ft = es_results['freetext']
es_pt = es_results['phrase']


#FREE TEXT EVALUATION

ft_precision = []
ft_recall = []
ft_fmeasure = []

for text in our_ft:
    
    if text != 'Avg Time':
    
        our_res = set(our_ft[text])
        es_res = set(es_ft[text])

        intersection = our_res & es_res

        tp = len(intersection)
        fp = len(our_res) - tp
        fn = len(es_res) - tp

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fmeasure = (2 * precision * recall) / (precision + recall)

        ft_precision.append(precision)
        ft_recall.append(recall)
        ft_fmeasure.append(fmeasure)
    
    else:

        ft_time = our_ft[text]

ft_precision_mean = statistics.mean(ft_precision)
ft_precision_med = statistics.median(ft_precision)

ft_recall_mean = statistics.mean(ft_recall)
ft_recall_med = statistics.median(ft_recall)

ft_fmeasure_mean = statistics.mean(ft_fmeasure)
ft_fmeasure_med = statistics.median(ft_fmeasure)


#PHRASE EVALUATION

pt_precision = []
pt_recall = []
pt_fmeasure = []

for text in our_pt:

    if text != 'Avg Time':
    
        our_res = set(our_pt[text])
        es_res = set(es_pt[text])

        intersection = our_res & es_res

        tp = len(intersection)
        fp = len(our_res) - tp
        fn = len(es_res) - tp

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fmeasure = (2 * precision * recall) / (precision + recall)

        pt_precision.append(precision)
        pt_recall.append(recall)
        pt_fmeasure.append(fmeasure)
    
    else:

        pt_time = our_pt[text]

pt_precision_mean = statistics.mean(pt_precision)
pt_precision_med = statistics.median(pt_precision)

pt_recall_mean = statistics.mean(pt_recall)
pt_recall_med = statistics.median(pt_recall)

pt_fmeasure_mean = statistics.mean(pt_fmeasure)
pt_fmeasure_med = statistics.median(pt_fmeasure)


print("FREETEXT EVALUATION")
print("Mean Precision: ",ft_precision_mean)
print("Mean Recall: ",ft_recall_mean)
print("Mean F-Measure: ",ft_fmeasure_mean)
print()
print("Median Precision: ",ft_precision_mean)
print("Median Recall: ",ft_recall_mean)
print("Median F-Measure: ",ft_fmeasure_mean)
print()
print("Average Time: ",ft_time)

print()
print()

print("PHRASE EVALUATION")
print("Mean Precision: ",pt_precision_mean)
print("Mean Recall: ",pt_recall_mean)
print("Mean F-Measure: ",pt_fmeasure_mean)
print()
print("Median Precision: ",pt_precision_mean)
print("Median Recall: ",pt_recall_mean)
print("Median F-Measure: ",pt_fmeasure_mean)
print()
print("Average Time: ",pt_time)



