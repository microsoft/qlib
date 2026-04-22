import Scale from './Scale.js';
import { ScaleTick, Dictionary } from '../util/types.js';
declare class IntervalScale<SETTING extends Dictionary<unknown> = Dictionary<unknown>> extends Scale<SETTING> {
    static type: string;
    type: string;
    protected _interval: number;
    protected _niceExtent: [number, number];
    private _intervalPrecision;
    parse(val: number): number;
    contain(val: number): boolean;
    normalize(val: number): number;
    scale(val: number): number;
    setExtent(start: number | string, end: number | string): void;
    unionExtent(other: [number, number]): void;
    getInterval(): number;
    setInterval(interval: number): void;
    /**
     * @param expandToNicedExtent Whether expand the ticks to niced extent.
     */
    getTicks(expandToNicedExtent?: boolean): ScaleTick[];
    getMinorTicks(splitNumber: number): number[][];
    /**
     * @param opt.precision If 'auto', use nice presision.
     * @param opt.pad returns 1.50 but not 1.5 if precision is 2.
     */
    getLabel(data: ScaleTick, opt?: {
        precision?: 'auto' | number;
        pad?: boolean;
    }): string;
    /**
     * @param splitNumber By default `5`.
     */
    calcNiceTicks(splitNumber?: number, minInterval?: number, maxInterval?: number): void;
    calcNiceExtent(opt: {
        splitNumber: number;
        fixMin?: boolean;
        fixMax?: boolean;
        minInterval?: number;
        maxInterval?: number;
    }): void;
    setNiceExtent(min: number, max: number): void;
}
export default IntervalScale;
