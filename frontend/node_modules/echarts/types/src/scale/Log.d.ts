import Scale from './Scale.js';
import IntervalScale from './Interval.js';
import SeriesData from '../data/SeriesData.js';
import { DimensionName, ScaleTick } from '../util/types.js';
declare class LogScale extends Scale {
    static type: string;
    readonly type = "log";
    base: number;
    private _originalScale;
    private _fixMin;
    private _fixMax;
    private _interval;
    private _niceExtent;
    /**
     * @param Whether expand the ticks to niced extent.
     */
    getTicks(expandToNicedExtent?: boolean): ScaleTick[];
    setExtent(start: number, end: number): void;
    /**
     * @return {number} end
     */
    getExtent(): [number, number];
    unionExtent(extent: [number, number]): void;
    unionExtentFromData(data: SeriesData, dim: DimensionName): void;
    /**
     * Update interval and extent of intervals for nice ticks
     * @param approxTickNum default 10 Given approx tick number
     */
    calcNiceTicks(approxTickNum: number): void;
    calcNiceExtent(opt: {
        splitNumber: number;
        fixMin?: boolean;
        fixMax?: boolean;
        minInterval?: number;
        maxInterval?: number;
    }): void;
    parse(val: any): number;
    contain(val: number): boolean;
    normalize(val: number): number;
    scale(val: number): number;
    getMinorTicks: IntervalScale['getMinorTicks'];
    getLabel: IntervalScale['getLabel'];
}
export default LogScale;
