#! /usr/bin/env python

import sys
import json

import numpy as np

from collections import defaultdict

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd

import seaborn as sns

outputPrefix = sys.argv[1]
fn = sys.argv[2]

def ReadDataByAdapter( fn ):
    byAdp = defaultdict(list)
    with open( fn ) as handle:
        columns = handle.next().strip().split(',')
        aTypeIdx  = columns.index("AdpType")
        cTypeIdx  = columns.index("CallType")
        zIdx      = columns.index("Zmw")
        zAccIdx   = columns.index("ZmwAccuracy")
        isRealIdx = columns.index("isReal")
        isHitIdx = columns.index("isHit")
        cAccIdx   = columns.index("CallAccuracy")
        for line in handle:
            row = line.strip().split(',')

            hn = int(row[zIdx].split('/')[-1])
            zAcc = float(row[zAccIdx])
            isReal = True if row[isRealIdx].lower() == "t" else False
            isHit = True if row[isHitIdx].lower() == "t" else False
            cAcc = float(row[cAccIdx])
            rec = (hn, zAcc, cAcc, isReal, isHit)

            # Only retain records with one unique adapter type
            tList = [t for t in [row[aTypeIdx], row[cTypeIdx]] if t != "-1"]
            if len(list(set(tList))) == 1:
                adp = tList[0]
                byAdp[adp].append( rec )
    return byAdp

def CallAccuracyPlots( name, data ):
    df = None
    for adp, vals in data.iteritems():
        filtered = [v for v in vals if v[2] > 0.0]
        TP = [v[2] for v in filtered if v[3]]
        FP = [v[2] for v in filtered if not v[3]]
        classes = (["TruePos"] * len(TP)) + (["FalsePos"] * len(FP))
        raw = {"AdapterType": adp,
               "AdapterClass": pd.Series(classes),
               "CallAccuracy": pd.Series(TP + FP)}

        if df is None:
            df = pd.DataFrame(raw)
        else:
            df = df.append(pd.DataFrame(raw))

    ax = sns.factorplot(x="AdapterType", y="CallAccuracy", hue="AdapterClass", kind="box", data=df)
    plt.subplots_adjust(top=0.9)
    ax.fig.suptitle("Adapter Call Accuracy by Type and Classification")

    plt.ylim(0.4, 1.05)
    pltFilename = "{0}_call_accuracy_box.png".format(name)
    plt.savefig(pltFilename)
    plt.close()

    p1 = {"caption": "Adapter Call Accuracy Box Plots For Adapter Types",
          "image": pltFilename,
          "tags": [],
          "id": "{0} - Adapter Call Accuracy Box Plots".format(name),
          "title": "{0} - Adapter Call Accuracy Box Plots".format(name),
          "uid": "0500001"}

    g = sns.FacetGrid(pd.melt(df,
                          id_vars=['AdapterType','AdapterClass'],
                          value_vars=['CallAccuracy']),
                          hue='AdapterClass', row='AdapterType', aspect=2.0)
    g.map(sns.kdeplot,'value', shade=True)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Adapter Call Accuracy by Type and Classification")
    plt.legend()
    pltFilename = "{0}_call_accuracy_dist.png".format(name)
    plt.savefig(pltFilename)
    plt.close()

    p2 = {"caption": "Adapter Call Accuracy Density Plot For Adapter Types",
          "image": pltFilename,
          "tags": [],
          "id": "{0} - Adapter Call Accuracy Density Plot".format(name),
          "title": "{0} - Adapter Call Accuracy Density Plot".format(name),
          "uid": "0500002"}

    g = sns.FacetGrid(pd.melt(df,
                          id_vars=['AdapterType','AdapterClass'],
                          value_vars=['CallAccuracy']),
                          hue='AdapterClass', row='AdapterType', aspect=2.0)
    bins = [x/1000.0 for x in range(400, 1001, 25)]
    g.map(plt.hist, 'value', alpha=0.5, bins=bins)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Adapter Call Accuracy by Type and Classification")
    plt.legend()
    pltFilename = "{0}_call_accuracy_hist.png".format(name)
    plt.savefig(pltFilename)

    p3 = {"caption": "Adapter Call Accuracy Histogram For Adapter Types",
          "image": pltFilename,
          "tags": [],
          "id": "{0} - Adapter Call Accuracy Histogram".format(name),
          "title": "{0} - Adapter Call Accuracy Histogram".format(name),
          "uid": "0500003"}

    return [p1, p2, p3]

def ZmwAccuracyPlots( name, data ):
    df = None
    for adp, vals in data.iteritems():
        filtered = [v for v in vals if v[3]]
        TP = [v[1] for v in filtered if v[4]]
        FN = [v[1] for v in filtered if not v[4]]
        classes = (["TruePos"] * len(TP)) + (["FalseNeg"] * len(FN))
        raw = {"AdapterType": adp,
               "AdapterClass": pd.Series(classes),
               "ZmwAccuracy": pd.Series(TP + FN)}

        if df is None:
            df = pd.DataFrame(raw)
        else:
            df = df.append(pd.DataFrame(raw))

    ax = sns.factorplot(x="AdapterType", y="ZmwAccuracy", hue="AdapterClass", kind="box", data=df)
    plt.subplots_adjust(top=0.9)
    ax.fig.suptitle("ZMW Accuracy for Adapter Calls\nby Type and Classification")

    plt.ylim(0.4, 1.05)
    pltFilename = "{0}_zmw_accuracy_box.png".format(name)
    plt.savefig(pltFilename)
    plt.close()

    p1 = {"caption": "ZMW Accuracy Box Plots For Adapter Types",
          "image": pltFilename,
          "tags": [],
          "id": "{0} - ZMW Accuracy Box Plots".format(name),
          "title": "{0} - ZMW Accuracy Box Plots".format(name),
          "uid": "0500004"}

    g = sns.FacetGrid(pd.melt(df,
                          id_vars=['AdapterType','AdapterClass'],
                          value_vars=['ZmwAccuracy']),
                          hue='AdapterClass', row='AdapterType', aspect=2.0)
    g.map(sns.kdeplot,'value', shade=True)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("ZMW Accuracy for Adapter Calls\nby Type and Classification")
    plt.legend()
    pltFilename = "{0}_zmw_accuracy_dist.png".format(name)
    plt.savefig(pltFilename)

    p2 = {"caption": "ZMW Accuracy Density Plot For Adapter Types",
          "image": pltFilename,
          "tags": [],
          "id": "{0} - ZMW Accuracy Density Plot".format(name),
          "title": "{0} - ZMW Accuracy Density Plot".format(name),
          "uid": "0500005"}

    g = sns.FacetGrid(pd.melt(df,
                          id_vars=['AdapterType','AdapterClass'],
                          value_vars=['ZmwAccuracy']),
                          hue='AdapterClass', row='AdapterType', aspect=2.0)
    bins = [x/1000.0 for x in range(400, 1001, 25)]
    g.map(plt.hist, 'value', alpha=0.5, bins=bins)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("ZMW Accuracy for Adapter Calls\nby Type and Classification")
    plt.legend()
    pltFilename = "{0}_zmw_accuracy_hist.png".format(name)
    plt.savefig(pltFilename)
    plt.close()

    p3 = {"caption": "ZMW Accuracy Histogram For Adapter Types",
          "image": pltFilename,
          "tags": [],
          "id": "{0} - ZMW Accuracy Histogram".format(name),
          "title": "{0} - ZMW Accuracy Histogram".format(name),
          "uid": "0500006"}

    return [p1, p2, p3]

def WriteReportJson( plotList=[], tableList=[] ):
    reportDict = {"plots":plotList, "tables":tableList}
    reportStr = json.dumps(reportDict, indent=1)
    with open("report.json", 'w') as handle:
        handle.write(reportStr)

data = ReadDataByAdapter( fn )
call_plots = CallAccuracyPlots( outputPrefix.lower(), data )
zmw_plots = ZmwAccuracyPlots( outputPrefix.lower(), data )
WriteReportJson( call_plots + zmw_plots )
