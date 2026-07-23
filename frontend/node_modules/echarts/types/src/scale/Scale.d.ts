import * as clazzUtil from '../util/clazz.js';
import { Dictionary } from 'zrender/lib/core/types.js';
import SeriesData from '../data/SeriesData.js';
import { DimensionName, ScaleDataValue, OptionDataValue, DimensionLoose, ScaleTick } from '../util/types.js';
import { ScaleRawExtentInfo } from '../coord/scaleRawExtentInfo.js';
declare abstract class Scale<SETTING extends Dictionary<unknown> = Dictionary<unknown>> {
    type: string;
    private _setting;
    protected _extent: [number, number];
    private _isBlank;
    readonly rawExtentInfo: ScaleRawExtentInfo;
    constructor(setting?: SETTING);
    getSetting<KEY extends keyof SETTING>(name: KEY): SETTING[KEY];
    /**
     * Parse input val to valid inner number.
     * Notice: This would be a trap here, If the implementation
     * of this method depends on extent, and this method is used
     * before extent set (like in dataZoom), it would be wrong.
     * Nevertheless, parse does not depend on extent generally.
     */
    abstract parse(val: OptionDataValue): number;
    /**
     * Whether contain the given value.
     */
    abstract contain(val: ScaleDataValue): boolean;
    /**
     * Normalize value to linear [0, 1], return 0.5 if extent span is 0.
     */
    abstract normalize(val: ScaleDataValue): number;
    /**
     * Scale normalized value to extent.
     */
    abstract scale(val: number): number;
    /**
     * Set extent from data
     */
    unionExtent(other: [number, number]): void;
    /**
     * Set extent from data
     */
    unionExtentFromData(data: SeriesData, dim: DimensionName | DimensionLoose): void;
    /**
     * Get extent
     *
     * Extent is always in increase order.
     */
    getExtent(): [number, number];
    /**
     * Set extent
     */
    setExtent(start: number, end: number): void;
    /**
     * If value is in extent range
     */
    isInExtentRange(value: number): boolean;
    /**
     * When axis extent depends on data and no data exists,
     * axis ticks should not be drawn, which is named 'blank'.
     */
    isBlank(): boolean;
    /**
     * When axis extent depends on data and no data exists,
     * axis ticks should not be drawn, which is named 'blank'.
     */
    setBlank(isBlank: boolean): void;
    /**
     * Update interval and extent of intervals for nice ticks
     *
     * @param splitNumber Approximated tick numbers. Optional.
     *        The implementation of `niceTicks` should decide tick numbers
     *        whether `splitNumber` is given.
     * @param minInterval Optional.
     * @param maxInterval Optional.
     */
    abstract calcNiceTicks(splitNumber?: number, minInterval?: number, maxInterval?: number): void;
    abstract calcNiceExtent(opt?: {
        splitNumber?: number;
        fixMin?: boolean;
        fixMax?: boolean;
        minInterval?: number;
        maxInterval?: number;
    }): void;
    /**
     * @return label of the tick.
     */
    abstract getLabel(tick: ScaleTick): string;
    abstract getTicks(): ScaleTick[];
    abstract getMinorTicks(splitNumber: number): number[][];
    static registerClass: clazzUtil.ClassManager['registerClass'];
    static getClass: clazzUtil.ClassManager['getClass'];
}
export default Scale;
