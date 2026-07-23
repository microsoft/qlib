declare const theme: {
    darkMode: boolean;
    color: string[];
    backgroundColor: string;
    axisPointer: {
        lineStyle: {
            color: string;
        };
        crossStyle: {
            color: string;
        };
        label: {
            color: string;
        };
    };
    legend: {
        textStyle: {
            color: string;
        };
        pageTextStyle: {
            color: string;
        };
    };
    textStyle: {
        color: string;
    };
    title: {
        textStyle: {
            color: string;
        };
        subtextStyle: {
            color: string;
        };
    };
    toolbox: {
        iconStyle: {
            borderColor: string;
        };
    };
    dataZoom: {
        borderColor: string;
        textStyle: {
            color: string;
        };
        brushStyle: {
            color: string;
        };
        handleStyle: {
            color: string;
            borderColor: string;
        };
        moveHandleStyle: {
            color: string;
            opacity: number;
        };
        fillerColor: string;
        emphasis: {
            handleStyle: {
                borderColor: string;
                color: string;
            };
            moveHandleStyle: {
                color: string;
                opacity: number;
            };
        };
        dataBackground: {
            lineStyle: {
                color: string;
                width: number;
            };
            areaStyle: {
                color: string;
            };
        };
        selectedDataBackground: {
            lineStyle: {
                color: string;
            };
            areaStyle: {
                color: string;
            };
        };
    };
    visualMap: {
        textStyle: {
            color: string;
        };
    };
    timeline: {
        lineStyle: {
            color: string;
        };
        label: {
            color: string;
        };
        controlStyle: {
            color: string;
            borderColor: string;
        };
    };
    calendar: {
        itemStyle: {
            color: string;
        };
        dayLabel: {
            color: string;
        };
        monthLabel: {
            color: string;
        };
        yearLabel: {
            color: string;
        };
    };
    timeAxis: {
        axisLine: {
            lineStyle: {
                color: string;
            };
        };
        splitLine: {
            lineStyle: {
                color: string;
            };
        };
        splitArea: {
            areaStyle: {
                color: string[];
            };
        };
        minorSplitLine: {
            lineStyle: {
                color: string;
            };
        };
    };
    logAxis: {
        axisLine: {
            lineStyle: {
                color: string;
            };
        };
        splitLine: {
            lineStyle: {
                color: string;
            };
        };
        splitArea: {
            areaStyle: {
                color: string[];
            };
        };
        minorSplitLine: {
            lineStyle: {
                color: string;
            };
        };
    };
    valueAxis: {
        axisLine: {
            lineStyle: {
                color: string;
            };
        };
        splitLine: {
            lineStyle: {
                color: string;
            };
        };
        splitArea: {
            areaStyle: {
                color: string[];
            };
        };
        minorSplitLine: {
            lineStyle: {
                color: string;
            };
        };
    };
    categoryAxis: {
        axisLine: {
            lineStyle: {
                color: string;
            };
        };
        splitLine: {
            lineStyle: {
                color: string;
            };
        };
        splitArea: {
            areaStyle: {
                color: string[];
            };
        };
        minorSplitLine: {
            lineStyle: {
                color: string;
            };
        };
    };
    line: {
        symbol: string;
    };
    graph: {
        color: string[];
    };
    gauge: {
        title: {
            color: string;
        };
        axisLine: {
            lineStyle: {
                color: (string | number)[][];
            };
        };
        axisLabel: {
            color: string;
        };
        detail: {
            color: string;
        };
    };
    candlestick: {
        itemStyle: {
            color: string;
            color0: string;
            borderColor: string;
            borderColor0: string;
        };
    };
};
export default theme;
