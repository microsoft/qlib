import Axis from './Axis.js';
import { AxisBaseModel } from './AxisBaseModel.js';
export declare function createAxisLabels(axis: Axis): {
    labels: {
        level?: number;
        formattedLabel: string;
        rawLabel: string;
        tickValue: number;
    }[];
    labelCategoryInterval?: number;
};
/**
 * @param {module:echats/coord/Axis} axis
 * @param {module:echarts/model/Model} tickModel For example, can be axisTick, splitLine, splitArea.
 * @return {Object} {
 *     ticks: Array.<number>
 *     tickCategoryInterval: number
 * }
 */
export declare function createAxisTicks(axis: Axis, tickModel: AxisBaseModel): {
    ticks: number[];
    tickCategoryInterval?: number;
};
/**
 * Calculate interval for category axis ticks and labels.
 * To get precise result, at least one of `getRotate` and `isHorizontal`
 * should be implemented in axis.
 */
export declare function calculateCategoryInterval(axis: Axis): number;
