/**
 * Base Axis Model for xAxis, yAxis, angleAxis, radiusAxis. singleAxis
 */
import { AxisBaseOptionCommon } from './axisCommonTypes.js';
import ComponentModel from '../model/Component.js';
import { AxisModelCommonMixin } from './axisModelCommonMixin.js';
import { AxisModelExtendedInCreator } from './axisModelCreator.js';
import Axis from './Axis.js';
export interface AxisBaseModel<T extends AxisBaseOptionCommon = AxisBaseOptionCommon> extends ComponentModel<T>, AxisModelCommonMixin<T>, AxisModelExtendedInCreator {
    axis: Axis;
}
