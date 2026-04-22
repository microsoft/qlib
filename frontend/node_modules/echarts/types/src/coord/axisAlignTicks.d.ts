import { NumericAxisBaseOptionCommon } from './axisCommonTypes.js';
import IntervalScale from '../scale/Interval.js';
import { AxisBaseModel } from './AxisBaseModel.js';
import LogScale from '../scale/Log.js';
export declare function alignScaleTicks(scale: IntervalScale | LogScale, axisModel: AxisBaseModel<Pick<NumericAxisBaseOptionCommon, 'min' | 'max'>>, alignToScale: IntervalScale | LogScale): void;
