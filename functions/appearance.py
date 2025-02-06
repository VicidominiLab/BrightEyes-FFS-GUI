def appearance(value):
    appearanceDic = {
        "subtlecol": "#C3C3C3", # subtle light grey color
        "strongcol": "#AEAEAE", # strong grey color for previous/next buttons
        "fbcol": "#DADADA",     # file button color
        "bgcol": "#FAFAFA",     # overall background color
        "linecol": "#D9D9D9",   # color vertical lines time trace
        "discol": "#FBA40A",    # color discarded data boxes
        "actbut": "#F5F6F7",    # color active file button
        "chunkbord": "#0084e8", # color border chunk time trace
        "colgrad": ['#3FCC14', '#4FBA2E', '#61A44D', '#6B985D', '#729168', '#7B8678', '#808080']
    }
    return appearanceDic[value]

# to do: derive short label from long label dictionary
def corr_short_label(corrMode):
    labelDic = {
        "Spot-variation fcs": "SpotVar",
        "All autocorrelations": "Autocorrs",
        "Pair-correlation fcs": "PairCorr",
        "STICS with iMSD analysis": "iMSD",
        "Free diff. 1 component": "free1C",
        "Anomalous diff. 1 component": "anom1C",
        "Free diff. 2 compoments": "free2C"
    }
    return labelDic[corrMode]

def corr_long_label(corrMode):
    labelDic = {
        "SpotVar": "Spot-variation fcs",
        "Autocorrs": "All autocorrelations",
        "PairCorr": "Pair-correlation fcs",
        "iMSD": "STICS with iMSD analysis",
        "free1C": "Free diff. 1 component",
        "anom1C": "Anomalous diff. 1 component",
        "free2C": "Free diff. 2 compoments",
    }
    if corrMode == "allValues":
        return list(labelDic.values())
    else:
        return labelDic[corrMode]