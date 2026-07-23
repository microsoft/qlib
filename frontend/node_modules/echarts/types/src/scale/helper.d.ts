import IntervalScale from './Interval.js';
import LogScale from './Log.js';
import Scale from './Scale.js';
declare type intervalScaleNiceTicksResult = {
    interval: number;
    intervalPrecision: number;
    niceTickExtent: [number, number];
};
export declare function isValueNice(val: number): boolean;
export declare function isIntervalOrLogScale(scale: Scale): scale is LogScale | IntervalScale;
/**
 * @param extent Both extent[0] and extent[1] should be valid number.
 *               Should be extent[0] < extent[1].
 * @param splitNumber splitNumber should be >= 1.
 */
export declare function intervalScaleNiceTicks(extent: [number, number], splitNumber: number, minInterval?: number, maxInterval?: number): intervalScaleNiceTicksResult;
export declare function increaseInterval(interval: number): number;
/**
 * @return interval precision
 */
export declare function getIntervalPrecision(interval: number): number;
export declare function fixExtent(niceTickExtent: [number, number], extent: [number, number]): void;
export declare function contain(val: number, extent: [number, number]): boolean;
export declare function normalize(val: number, extent: [number, number]): number;
export declare function scale(val: number, extent: [number, number]): number;
export {};
