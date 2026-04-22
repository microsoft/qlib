import { TimeUnit } from '../util/time.js';
import IntervalScale from './Interval.js';
import { TimeScaleTick } from '../util/types.js';
import { TimeAxisLabelFormatterOption } from '../coord/axisCommonTypes.js';
import { LocaleOption } from '../core/locale.js';
import Model from '../model/Model.js';
declare type TimeScaleSetting = {
    locale: Model<LocaleOption>;
    useUTC: boolean;
};
declare class TimeScale extends IntervalScale<TimeScaleSetting> {
    static type: string;
    readonly type = "time";
    _approxInterval: number;
    _minLevelUnit: TimeUnit;
    constructor(settings?: TimeScaleSetting);
    /**
     * Get label is mainly for other components like dataZoom, tooltip.
     */
    getLabel(tick: TimeScaleTick): string;
    getFormattedLabel(tick: TimeScaleTick, idx: number, labelFormatter: TimeAxisLabelFormatterOption): string;
    /**
     * @override
     */
    getTicks(): TimeScaleTick[];
    calcNiceExtent(opt?: {
        splitNumber?: number;
        fixMin?: boolean;
        fixMax?: boolean;
        minInterval?: number;
        maxInterval?: number;
    }): void;
    calcNiceTicks(approxTickNum: number, minInterval: number, maxInterval: number): void;
    parse(val: number | string | Date): number;
    contain(val: number): boolean;
    normalize(val: number): number;
    scale(val: number): number;
}
export default TimeScale;
